# 🤖 Leetcode Automata
本项目灵感来源 [lang-or-llama](https://github.com/unclefomotw/lang-or-llama), graphql来源 [alfa-leetcode-api](https://github.com/alfaarghya/alfa-leetcode-api), 旨在学习LangChain框架.

## 🤔 这是什么
本项目可以实现LeetCode做题流程的自动化 ✨

## ⚙️ 如何运作
本项目主要由以下3个步骤构成: 拉取题目, 编码, 提交代码.

### 📥 S1 拉取题目
Agent 拉取指定LeetCode题目.

### 💻 S2 编码
在这一步骤中, Agent负责对LeetCode题目进行解析, 生成测试用例, 生成主代码, 并在测试用例上对主代码进行测试.

### 🚀 S3 提交代码
Agent将完整的执行代码提交到LeetCode并确认最终执行结果.

## 📊 详细流程
参考流程图:

![](asset/image.png)

## 🔧 准备
1. 🍪 将用户的LeetCode cookie数据填写到leetcode_cookie.txt文件中.
2. 🔑 将.env文件中的OPENAI_API_KEY填写为有效的api key.

## 🎯 执行
```bash
pip install -r requirements.txt

## 📅 处理每日一题
python main.py --daily

## 🎯 处理指定题目
python main.py --title_slug=median-of-two-sorted-arrays
```

## 📋 待办项
- [ ] **用例执行**和**编码反思**流程粒度较粗, 反思效果非常有限, 需要将**用例执行**和**编码反思**细化到单个用例维度, 观测是否有助于提高反思效果. 🤖💭
