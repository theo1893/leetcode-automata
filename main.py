import argparse
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

from leetcode_api import fetch_daily_question, CodeSnippet, Question, submit_code
from prompt import (
    MAIN_CODE_GEN_SYSTEM_PROMPT,
    MAIN_CODE_GEN_USER_PROMPT,
    REGEN_BY_COMMENT_USER_PROMPT,
    REGEN_BY_ERROR_USER_PROMPT,
    PARSE_LEETCODE_PROBLEM_PROMPT, GENERATE_ASSERTION_PROMPT,
)
from util import (
    combine_test_with_code,
    extract_code,
    print_node_output, multiline_input
)

AI_TEST_REVISION_LIMIT = 2  ## ai test版本上限
AI_MAIN_REVISION_LIMIT = 6  ## ai main版本上限


class CodeExecutionResult(BaseModel):
    code: str
    stdout: str
    stderr: str
    has_error: bool


def execute_code(code: str) -> CodeExecutionResult:
    ## 在本地执行代码
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
    ## ai测试用例是否合理, 由ai自己判断
    is_ai_test_code_good: bool
    ## 是否跳过ai测试用例
    skip_ai_test_code: bool

    ## 上一轮执行结果
    last_test_result: CodeExecutionResult
    ## 上一轮执行使用的测试用例类型
    last_test_type: TestType

    has_human_interference: bool
    ## 人类对主代码的comment
    human_comment_on_main_code: str
    ## 人类补充的测试用例
    human_test_code: Annotated[str, lambda a, b: f"{a}\n{b}"]


class AITestResp(TypedDict):
    test_codes: str  ## 生成的测试代码
    thoughts: str  ## 思考?
    conclusion: str  ## 结论?


def get_codegen_workflow() -> StateGraph:
    _llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.001)


    def node_fetch_daily_question(state: AgentState):
        print("\n=== 开始拉取每日一题 ===\n")
        daily_question = fetch_daily_question()

        return {
            "raw_question": daily_question.data.activeDailyCodingChallengeQuestion.question,
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

    def node_generate_assertion(state: AgentState):
        print("\n=== 开始生成用例断言 ===\n")

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
            thoughts: str  ## 思考?
            conclusion: str  ## 结论?

        _system_message = SystemMessage(content=MAIN_CODE_GEN_SYSTEM_PROMPT)
        _local_llm = (lambda messages: [_system_message] + messages) | _llm.with_structured_output(MainCodeResp)

        problem = state["problem"]
        revision = state.get("main_code_revision", 0)

        if (not state.get("main_code")) or (not state.get("last_test_result")):
            # No reflection since no main code is generated, or no (valid) tests were performed
            message = HumanMessage(
                content=MAIN_CODE_GEN_USER_PROMPT.format(
                    problem_description=problem["problem_description"],
                    example_description=problem["example_description"],
                    solution_interface=problem["solution_interface"]
                ))
            resp = _local_llm.invoke([message])
        else:
            # Re-generate main code; reflect the previous error
            history_messages = state["main_coding_llm_messages"]
            last_test_result = state["last_test_result"]
            human_comment = state.get("human_comment_on_main_code", "")
            if last_test_result.has_error:
                message = HumanMessage(
                    content=REGEN_BY_ERROR_USER_PROMPT.format(
                        code=last_test_result.code,
                        error=last_test_result.stderr,
                        human_comment=human_comment
                    ))
            else:
                message = HumanMessage(
                    content=REGEN_BY_COMMENT_USER_PROMPT.format(
                        human_comment=human_comment
                    ))
            resp = _local_llm.invoke(history_messages + [message])

        revision += 1

        if not resp:
            code = None
        else:
            code = resp.get("main_codes", "")

        return {
            "main_coding_llm_messages": [message, AIMessage(json.dumps(resp))],
            "main_code": code,
            "main_code_revision": revision
        }

    def node_test_main_with_examples(state: AgentState):
        print("\n=== 开始执行测试用例 ===\n")

        main_code = state["main_code"]
        if not main_code:
            return {"is_main_code_good": False}

        example_test_code = state["problem"]["example_test_code"]

        if state["human_test_code"]:
            example_test_code += "\n" + state["human_test_code"]

        code_result = execute_code(
            combine_test_with_code(main_code, example_test_code)
        )
        return {
            "is_main_code_good": not code_result.has_error,
            "last_test_result": code_result,
            "last_test_type": TestType.EXAMPLES
        }

    def node_give_answer(state: AgentState):
        print("\n=== 结果 ===\n")

        print(">>>> SOLUTION <<<<")
        print(state["main_code"])
        print(">> SOLUTION END <<\n\n")

        if state["is_main_code_good"]:
            print("This code should work.  Please verify.")
        else:
            print(f"This code didn't pass the test from {state["last_test_type"]}:\n")
            print(state["last_test_result"].stderr, end="\n\n")

        if state.get("skip_ai_test_code", False):
            print("BTW, I tried AI-generated tests but skipped, because the tests themselves have problems.\n")

    def node_finalize(state: AgentState):
        print("\n=== 提交代码 ===\n")
        submit_code(state["raw_question"].titleSlug, state["raw_question"].questionId, state["main_code"])


    def to_regenerate_main_code(state: AgentState) -> str:
        revision = state.get("main_code_revision") or 0
        if revision >= AI_MAIN_REVISION_LIMIT:
            return "no"

        if state.get("is_main_code_good", False):
            return "no"
        else:
            return "yes"

    def has_human_interference(state: AgentState) -> str:
        if state.get("has_human_interference"):
            return "yes"
        else:
            return "no"

    workflow = StateGraph(AgentState)

    workflow.add_node("fetch_daily_question", node_fetch_daily_question)
    workflow.add_node("node_parse_leetcode_problem", node_parse_leetcode_problem)
    workflow.add_node("node_generate_assertion", node_generate_assertion)
    workflow.add_node("generate_main_code", node_generate_main_code)
    workflow.add_node("test_with_examples", node_test_main_with_examples)
    workflow.add_node("give_answer", node_give_answer)
    workflow.add_node("finalize", node_finalize)

    workflow.set_entry_point("fetch_daily_question")

    workflow.add_edge("fetch_daily_question", "node_parse_leetcode_problem")
    workflow.add_edge("node_parse_leetcode_problem", "node_generate_assertion")
    workflow.add_edge("node_generate_assertion", "generate_main_code")
    workflow.add_edge("generate_main_code", "test_with_examples")
    workflow.add_conditional_edges(
        source="test_with_examples",
        path=to_regenerate_main_code,
        path_map={
            "yes": "generate_main_code",
            "no": "give_answer"
        }
    )

    workflow.add_conditional_edges(
        source="give_answer",
        path=has_human_interference,
        path_map={
            "yes": "generate_main_code",
            "no": "finalize"
        }
    )
    workflow.add_edge("finalize", END)

    # from IPython.display import Image
    # with open('new_arch.png', 'wb') as f:
    #     f.write(Image(workflow.compile().get_graph().draw_mermaid_png()).data)

    return workflow


def solve_daily_question():
    config = {"configurable": {"thread_id": "user-24601-conv-1337"}}

    graph = get_codegen_workflow().compile(
        checkpointer=MemorySaver(),
        interrupt_after=["give_answer"]
    )

    initial_state = {}
    for s in graph.stream(initial_state, config=config):
        print_node_output(s)  # TODO: verbose

    # XXX: This pattern only works when the upcoming state at the interruption
    #      is not END, so we can distinguish END from interruption.
    #      Another pattern is to use `while True:` with conditional break by user_input
    while graph.get_state(config).next:
        user_test = multiline_input("Add more asserting tests if necessary")
        user_comment = input('Provide feedback to revise the code (just enter if none):')

        if user_test or user_comment:
            state = graph.get_state(config).values

            graph.update_state(
                config=config,
                values={
                    "has_human_interference": True,
                    "human_comment_on_main_code": user_comment,
                    "human_test_code": user_test,
                    "main_code_revision_limit": AI_MAIN_REVISION_LIMIT + 2
                }
            )
        else:
            graph.update_state(
                config=config,
                values={
                    "human_comment_on_main_code": "",
                    "has_human_interference": False
                }
            )

        for s in graph.stream(None, config=config):
            print_node_output(s)  # TODO: verbose


if __name__ == '__main__':
    load_dotenv()
    solve_daily_question()

