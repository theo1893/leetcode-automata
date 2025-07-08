import time
from typing import Optional, List

import requests
from pydantic import BaseModel
from rich import print

from graphql import DAILY_QUESTION_QUERY, QUESTION_DETAIL_QUERY


class CodeSnippet(BaseModel):
    lang: str
    langSlug: str
    code: str  ## 接口代码


class Question(BaseModel):
    questionId: str  ## 问题id
    titleSlug: str  ## url slug
    content: str  ## 原始内容 in html, 包括description + test cases
    codeSnippets: List[CodeSnippet]  ## 接口

class ActiveDailyCodingChallengeQuestion(BaseModel):
    question: Question

class DailyChallengeDetail(BaseModel):
    activeDailyCodingChallengeQuestion: ActiveDailyCodingChallengeQuestion

class DailyQuestionResp(BaseModel):
    data: DailyChallengeDetail

class DetailQuestionData(BaseModel):
    question: Question

class DetailQuestionResp(BaseModel):
    data: DetailQuestionData


class SubmitResp(BaseModel):
    submission_id: int  ## 提交id


class SubmissionStatus(BaseModel):
    finished: Optional[bool] = False  ## 是否完成
    status_runtime: Optional[str] = ""  ## 时间消耗
    status_memory: Optional[str] = ""  ## 内存消耗
    status_msg: Optional[str] = ""  ## 状态信息; Accepted - 成功
    input_formatted: Optional[str] = "" ## 失败的case

LEETCODE_GRAPHQL_URL = "https://leetcode.com/graphql"


## 获取每日一题
def fetch_daily_question() -> Question:
    data = {
        "query": DAILY_QUESTION_QUERY,
        "variables": {}
    }

    response = requests.post(LEETCODE_GRAPHQL_URL, json=data)
    daily_question = DailyQuestionResp.model_validate(response.json())

    return daily_question.data.activeDailyCodingChallengeQuestion.question

## 获取指定题目
def fetch_selected_question(title_slug: str) -> Question:
    data = {
        "query": QUESTION_DETAIL_QUERY,
        "variables": {
            "titleSlug": title_slug,
        }
    }

    response = requests.post(LEETCODE_GRAPHQL_URL, json=data)
    question_detail = DetailQuestionResp.model_validate(response.json())
    return question_detail.data.question


## 提交代码
def submit_code(title_slug: str, question_id: str, code: str) -> SubmissionStatus:
    f = open('leetcode_cookie.txt', 'r')
    cookies = {}
    for line in f.read().split(';'):
        name, value = line.strip().split('=', 1)
        cookies[name] = value

    url = f"https://leetcode.cn/problems/{title_slug}/submit"
    data = {
        "lang": "python3",
        "question_id": question_id,
        "typed_code": code,
    }

    response = requests.post(url, json=data, cookies=cookies)

    submit_resp = SubmitResp.model_validate(response.json())

    print(f"提交成功:\n{submit_resp}")

    print("开始轮询提交状态...")
    return check_submission_status(submit_resp.submission_id)



## 轮询状态
def check_submission_status(submission_id) -> SubmissionStatus:
    url = f"https://leetcode.cn/submissions/detail/{submission_id}/check/"

    while True:
        response = requests.get(url)
        check_resp = SubmissionStatus.model_validate(response.json())

        if not check_resp.finished:
            print(f"执行中...")
            time.sleep(1)
            continue

        print("执行完成")
        print(check_resp)
        return check_resp


if __name__ == '__main__':
    # fetch_selected_question("maximum-number-of-events-that-can-be-attended-ii")
    submit_code("maximum-number-of-events-that-can-be-attended-ii", "1851", '''from typing import List
import bisect
import time

class Solution:
    def maxValue(self, events: List[List[int]], k: int) -> int:
        while True:
            time.sleep(1)''')
