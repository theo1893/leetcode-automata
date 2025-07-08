import json
import json
import os
import subprocess
import tempfile
from enum import Enum
from traceback import print_exc
from typing import TypedDict, Annotated, List

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from rich import print

from leetcode_api import fetch_daily_question, Question, submit_code, SubmissionStatus
from prompt import (
    MAIN_CODE_GEN_SYSTEM_PROMPT,
    MAIN_CODE_GEN_USER_PROMPT,
    REGEN_BY_ERROR_USER_PROMPT,
    PARSE_LEETCODE_PROBLEM_PROMPT, GENERATE_ASSERTION_PROMPT, APPEND_ASSERTION_SYSTEM_PROMPT,
    APPEND_ASSERTION_USER_PROMPT,
)
from util import (
    combine_test_with_code
)

AI_MAIN_REVISION_LIMIT = 6  ## ai main版本上限


class CodeExecutionResult(BaseModel):
    code: str
    stdout: str
    stderr: str
    has_error: bool


## 在本地执行代码
def execute_code(code: str) -> CodeExecutionResult:
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(code.encode('utf-8'))
        tmp_file_path = f.name

    try:
        result = subprocess.run(
            ["python", tmp_file_path],
            capture_output=True,
            timeout=5,
            check=False,
            encoding="utf-8"
        )

        os.remove(tmp_file_path)

        return CodeExecutionResult(
            code=code,
            stdout=result.stdout,
            stderr=result.stderr,
            has_error=(result.returncode != 0)
        )
    except Exception as e:
        print_exc()
        return CodeExecutionResult(
            code=code,
            stdout="",
            stderr=f"{e}",
            has_error=True
        )


class TestType(Enum):
    AI = 1
    EXAMPLES = 2

    def __str__(self):
        return self.name


class LeetCodeProblem(TypedDict):
    problem_description: str
    example_description: str
    solution_interface: str
    example_test_code: str


class Example(TypedDict):
    input: str
    output: str
    explanation: str

class ParsedQuestion(TypedDict):
    description: str
    examples: List[Example]

class AgentState(TypedDict):
    ## 消息历史
    main_coding_llm_messages: Annotated[list[BaseMessage], add_messages]

    ## 原始数据
    raw_question: Question
    parsed_question_description: str
    parsed_examples: List[Example]
    parsed_examples_str: str
    parsed_interface_str: str
    generated_test_cases: str


    ## leetcode 题目描述
    problem: LeetCodeProblem

    ## ai生成的主代码
    main_code: str
    ## ai主代码版本
    main_code_revision: int

    ## 主代码是否有效, 根据执行结果判断
    is_main_code_good: bool

    ## 上一轮本地测试执行结果
    last_test_result: CodeExecutionResult
    ## 上一轮提交执行结果
    last_submission_result: SubmissionStatus


def get_codegen_workflow() -> StateGraph:
    _llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.001)


    def node_fetch_daily_question(state: AgentState):
        print("\n=== 开始拉取每日一题 ===\n")
        daily_question = fetch_daily_question()

        return {
            "raw_question": daily_question,
        }

    def node_parse_leetcode_problem(state: AgentState):
        print("\n=== 开始解析题目 ===\n")

        _system_message = SystemMessage(content=PARSE_LEETCODE_PROBLEM_PROMPT)
        _local_llm = (lambda messages: [_system_message] + messages) | _llm.with_structured_output(ParsedQuestion)

        message = HumanMessage(content=state["raw_question"].content)
        resp = _local_llm.invoke(input=[message])

        parsed_examples_str = ""
        for case in resp.get("examples", []):
            parsed_examples_str  = parsed_examples_str + f"Input:\n{case['input']}\n"
            parsed_examples_str  = parsed_examples_str + f"Output:\n{case['output']}\n"
            if case.get('explanation') is not None:
                parsed_examples_str  = parsed_examples_str + f"Explanation:\n{case['explanation']}\n"
            parsed_examples_str = parsed_examples_str + "\n"

        parsed_interface_str = ""
        for interface in state['raw_question'].codeSnippets:
            if interface.langSlug == 'python3':
                parsed_interface_str = interface.code
                break

        return {
            "parsed_question_description": resp.get("description", ""),
            "parsed_examples": resp.get("examples", []),
            "parsed_examples_str": parsed_examples_str,
            "parsed_interface_str": parsed_interface_str,
        }

    ## TODO 把提交有误的case加进去
    def node_generate_assertion(state: AgentState):
        print("\n=== 开始生成用例断言 ===\n")

        ## 提交失败后重新生成case的路径
        if state.get("last_submission_result", None) is not None:
            _system_message = SystemMessage(content=APPEND_ASSERTION_SYSTEM_PROMPT)
            _local_llm = (lambda messages: [_system_message] + messages) | _llm
            _user_message = HumanMessage(content=APPEND_ASSERTION_USER_PROMPT.format(
                stardard_cases=state["generated_test_cases"],
                new_case=state["last_submission_result"].input_formatted,
            ))
            resp = _local_llm.invoke(input=[_user_message])
        else:
            _system_message = SystemMessage(content=GENERATE_ASSERTION_PROMPT)
            _local_llm = (lambda messages: [_system_message] + messages) | _llm

            content = ""
            content = content + f"Cases:\n```{state['parsed_examples_str']}```\n\n"
            content = content + f"Interface:\n```{state['parsed_interface_str']}```"

            message = HumanMessage(content=content)
            resp = _local_llm.invoke(input=[message])

        leetcode_problem = LeetCodeProblem(
            problem_description=state['parsed_question_description'],
            example_description=state['parsed_examples_str'],
            solution_interface=state['parsed_interface_str'],
            example_test_code=resp.content,
        )

        return {
            "generated_test_cases": resp,
            "problem": leetcode_problem,
        }


    def node_generate_main_code(state: AgentState):
        print("\n=== 开始生成主代码 ===\n")

        class MainCodeResp(TypedDict):
            main_codes: str  ## 生成的主代码
            thoughts: str  ## 思考

        _system_message = SystemMessage(content=MAIN_CODE_GEN_SYSTEM_PROMPT)
        _local_llm = (lambda messages: [_system_message] + messages) | _llm.with_structured_output(MainCodeResp)

        problem = state["problem"]
        revision = state.get("main_code_revision", 0)

        if (not state.get("main_code")) or (not state.get("last_test_result")):
            ## 第一次进来, 生成全新的主代码
            message = HumanMessage(
                content=MAIN_CODE_GEN_USER_PROMPT.format(
                    problem_description=problem["problem_description"],
                    example_description=problem["example_description"],
                    solution_interface=problem["solution_interface"]
                ))
            resp = _local_llm.invoke([message])
        else:
            ## 非第一次进来, 需要reflection
            history_messages = state["main_coding_llm_messages"]
            last_test_result = state["last_test_result"]

            ## TODO 把所有case丢进去
            if last_test_result.has_error:
                ## 上一轮本地跑的case报错, 重新生成主代码
                message = HumanMessage(
                    content=REGEN_BY_ERROR_USER_PROMPT.format(
                        code=state.get("main_code", ""),
                        error=last_test_result.stderr,
                    ))
            else:
                ## 上一轮提交结果有误, 重新生成主代码
                message = HumanMessage(
                    content=REGEN_BY_ERROR_USER_PROMPT.format(
                        code=state.get("main_code", ""),
                        error=state["last_submission_result"].status_msg,
                    )
                )
            resp = _local_llm.invoke(history_messages + [message])

        revision += 1

        return {
            "main_coding_llm_messages": [message, AIMessage(json.dumps(resp))],
            "main_code": resp.get("main_codes", ""),
            "main_code_revision": revision
        }

    def node_test_main_with_examples(state: AgentState):
        print("\n=== 开始执行测试用例 ===\n")

        main_code = state["main_code"]
        if not main_code:
            return {"is_main_code_good": False}

        example_test_code = state["problem"]["example_test_code"]

        code_result = execute_code(
            combine_test_with_code(main_code, example_test_code)
        )
        return {
            "is_main_code_good": not code_result.has_error,
            "last_test_result": code_result,
        }

    def node_give_answer(state: AgentState):
        print("\n=== 测试结果 ===\n")

        print(">>>> 主代码 <<<<")
        print(state["main_code"])
        print(">> 主代码 <<")

        if state["is_main_code_good"]:
            print("生成代码通过用例, 准备提交.")
        else:
            print(f"生成代码未通过用例, 报错: {state["last_test_result"].stderr}")


    def node_submit(state: AgentState):
        print("\n=== 提交代码 ===\n")
        submission_status = submit_code(state["raw_question"].titleSlug, state["raw_question"].questionId, state["main_code"])

        return {
            "last_submission_result": submission_status,
            "is_main_code_good": True if submission_status.status_msg == "Accepted" else False,
        }


    def to_regenerate_main_code(state: AgentState) -> str:
        revision = state.get("main_code_revision") or 0
        if revision >= AI_MAIN_REVISION_LIMIT:
            return "no-regen"

        if state.get("is_main_code_good", False):
            return "no-regen"
        else:
            return "regen"

    def to_regenerate_testcase(state: AgentState) -> str:
        revision = state.get("main_code_revision") or 0
        if revision >= AI_MAIN_REVISION_LIMIT:
            return "no-regen"

        if state.get("is_main_code_good", False):
            return "no-regen"
        else:
            return "regen"


    workflow = StateGraph(AgentState)

    workflow.add_node("fetch_daily_question", node_fetch_daily_question)
    workflow.add_node("node_parse_leetcode_problem", node_parse_leetcode_problem)
    workflow.add_node("node_generate_assertion", node_generate_assertion)
    workflow.add_node("generate_main_code", node_generate_main_code)
    workflow.add_node("test_with_examples", node_test_main_with_examples)
    workflow.add_node("give_answer", node_give_answer)
    workflow.add_node("submit", node_submit)

    workflow.set_entry_point("fetch_daily_question")

    workflow.add_edge("fetch_daily_question", "node_parse_leetcode_problem")
    workflow.add_edge("node_parse_leetcode_problem", "node_generate_assertion")
    workflow.add_edge("node_generate_assertion", "generate_main_code")
    workflow.add_edge("generate_main_code", "test_with_examples")
    workflow.add_edge("give_answer", "submit")
    workflow.add_conditional_edges(
        source="test_with_examples",
        path=to_regenerate_main_code,
        path_map={
            "regen": "generate_main_code",
            "no-regen": "give_answer",
        }
    )
    workflow.add_conditional_edges(
        source="submit",
        path=to_regenerate_testcase,
        path_map={
            "regen": "node_generate_assertion",
            "no-regen": END
        }
    )

    # from IPython.display import Image
    # with open('new_arch.png', 'wb') as f:
    #     f.write(Image(workflow.compile().get_graph().draw_mermaid_png()).data)

    return workflow


def solve_daily_question():
    config = {"configurable": {"thread_id": "theo-1893"}}

    graph = get_codegen_workflow().compile(
        checkpointer=MemorySaver(),
    )

    initial_state = {}
    for s in graph.stream(initial_state, config=config):
        print(s)


if __name__ == '__main__':
    load_dotenv()
    solve_daily_question()

