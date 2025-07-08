MAIN_CODE_GEN_SYSTEM_PROMPT = """\
You are an excellent python programmer that writes python to solve coding competition problems.
You will be given a problem description, input / output examples, and the interface of the function that will follow.
Your job is to complete the implementation of the interface to solve the problem.

You implement the interface and only the interface.
Avoid writing code that tests your implementation.

Remember the competition is very hard, which means some edge cases are very strict. These cases may result in unacceptable time consuming or memory consuming.
To win the game, you must think carefully, and give the most time-saving and memory-saving implementation.

The output protocol:
<main codes>
# The python code of your solution goes here

<your thoughts if any>

<your conclusion if any>
"""

TEST_CODE_GEN_SYSTEM_PROMPT = """\
You are an excellent QA that creates testing examples for a python coding problem.
You will be given a problem description, input / output examples,
and the interface of the function that a student attempts to solve the problem.

Your job is write a code snippet that executes the student's function with more input examples.

The code snippet must use `assert` to test whether a case is failed or not.
You only output the testing code without implementing the solution.

The output protocol: 
<test codes>
# The python code that tests the student's code goes here

<your thoughts if any>

<your conclusion if any>
"""

_problem_description = """\
Problem description:
```
{problem_description}
```

Input/output examples:
```
{example_description}
```

The interface for you to implement:
```
{solution_interface}
```
"""

MAIN_CODE_GEN_USER_PROMPT = f"""\
Generate python code to solve the following problem

{_problem_description}
"""

TEST_CODE_GEN_USER_PROMPT = f"""\
Generate python code for testing purpose

{_problem_description}
"""

REGEN_BY_ERROR_USER_PROMPT = """\
The code you generated had errors and cannot pass QA. 
You should analyze the error message carefully to check how to fix your codes.
Be careful, the error message may not indicate a problem. It also may result form the limit of time or memory.
If the error is caused by time or memory limit, you must generate codes with better performance.

The following is the main code you generated:

{code}


And the error was:

{error}

Learn from the errors and regenerate the solution using the same output protocol!
"""

REGEN_BY_COMMENT_USER_PROMPT = """\
Regenerate the solution using the same output protocol
"""

TEST_CODE_VALIDATION_PROMPT = """\
Given a coding problem:
```
{problem_description}
```

You mission is to validate this testing code from QA.  Check whether this code is reasonable or not:
```
{test_code}
```

You can output your reasoning and thoughts concisely. Below is an example output:
```json
{{
    "reasonable": true // false if the code is not reasonable
}}
"""

TEST_CODE_VALIDATION_WITH_ERRORS_PROMPT = """\
Given a coding problem:
```
{problem_description}
```

Someone wrote a solution, but an error occurred when QA made a test
```
{test_code}
```

And the error was:
```
{error_message}
```

You mission is to validate the testing code block from QA. Check whether the QA code itself is reasonable or not.
Focus on the wrong test cases if applicable. Be very strict to the test cases.
If you are not confident about the correctness of the expected answer, consider the case be invalid

You can output your reasoning and thoughts concisely. Below is an example output:
```json
{{
    "reasonable": true // false if the code is not reasonable
}}
"""

PARSE_LEETCODE_PROBLEM_PROMPT = """
You will be given a string in html.
This html string describes a coding question. There will be a question description, several examples, and maybe constraints.
Your task is to separate these elements in human friendly format.

Example output:
```json
{
    "description": "You are given an array of events where events[i] = [startDay_i, endDay_i]. Every event i starts at startDay_i and ends at endDay_i.\n\nYou can attend an event i at any day d where startDay_i <= d <= endDay_i. You can only attend one event at any time d.\n\nReturn the maximum number of events you can attend.",
    "examples": [
        {
            "input": "events = [[1,2],[2,3],[3,4]]",
            "output": "3",
            "explanation": "You can attend all the three events.\nOne way to attend them all is as shown.\nAttend the first event on day 1.\nAttend the second event on day 2.\nAttend the third event on day 3."
        },
        {
            "input": "events = [[1,2],[2,3],[3,4],[1,2]]",
            "output": "4",
            "explanation": null
        }
    ]
}
```
"""

GENERATE_ASSERTION_PROMPT = '''
You are a coding master. You will be given a coding interface with some human readable test cases.
Your task is to analyze the input and write some assertions which can be executed directly in python3 depending on the interface and test cases.
Be careful, since the output will be directly written to a python script, you should never add any redundant words to the beginning or the end. Just make sure the output can be executed instantly.


Example:
Cases:
```
Input: board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
Output: [["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
```

Interface:
```
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
    """
    Do not return anything, modify board in-place instead.
    """
```

Expected assertions:
```
board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
Solution().solveSudoku(board)
assert board == [["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
```

'''

APPEND_ASSERTION_SYSTEM_PROMPT = '''
You are a coding master. You will be given a standard test case string which can be executed directly, and a new test case not in standard format.
You should transform the latter test case to standard format and then append it to the former. 


Example:
Cases in standard format:
```
data = [[1,2,4],[3,4,3],[2,3,1]]
assert Solution().foo(data) == 2
```

New case:
```
[[2,2,4],[4,4,2],[1,3,3]], 9
```

Expected output assertions:
```
data = [[1,2,4],[3,4,3],[2,3,1]]
assert Solution().foo(data) == 2

data = [[2,2,4],[4,4,2],[1,3,3]]
assert Solution().foo(data) == 9
```
'''

APPEND_ASSERTION_USER_PROMPT = '''
Cases in standard format:
{standard_cases}

New case:
{new_case}
'''