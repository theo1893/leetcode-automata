import time
from typing import Optional, List, TypedDict

import requests
from rich import print

from pydantic import BaseModel

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


class SubmitResp(BaseModel):
    submission_id: int  ## 提交id


class CheckResp(BaseModel):
    finished: Optional[bool] = False  ## 是否完成
    run_success: Optional[bool] = False  ## 是否pass
    status_runtime: Optional[str] = ""  ## 时间消耗
    status_memory: Optional[str] = ""  ## 内存消耗


## 获取每日一题
def fetch_daily_question() -> DailyQuestionResp:
    response = requests.get(f"http://localhost:3000/dailyQuestion")
    daily_question = DailyQuestionResp.model_validate(response.json())

    return daily_question


## 提交代码
def submit_code(title_slug: str, question_id: str, code: str):
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
    check_submission(submit_resp.submission_id)


## 轮询状态
def check_submission(submission_id):
    url = f"https://leetcode.cn/submissions/detail/{submission_id}/check/"

    while True:
        response = requests.get(url)
        check_resp = CheckResp.model_validate(response.json())

        if not check_resp.finished:
            print(f"执行中...")
            time.sleep(1)
            continue

        print("执行完成")
        print(check_resp)
        break


if __name__ == '__main__':
    fetch_daily_question()
