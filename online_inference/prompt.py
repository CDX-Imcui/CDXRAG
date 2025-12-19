SYSTEM_EXPLORE_PROMPT = """Next, you will complete a table-related question answering task. Based on the provided materials such as the table content (in Markdown format), you need to analyze the User Query.
And try to decide whether the User Input Query should be broken down into subqueries. You are provided with "solve_subquery" tool that can get answer for the subqueries.
After you have collected sufficient information, you need to generate comprehensive answers.

Table Contet: {table_content}

Instructions:
1. Carefully analyze each user query through step-by-step reasoning.
2. If the query needs information more than the given table content：
    - Decompose the query into subqueries.
    - Process one subquery at a time.
    - Use "solve_subquery" tool to get answers for each subquey.
3. If a query can be answered by table content, do not decompose it. And directly put the orignal query into the "solve_subquery" tool.
    The "solve_subquery" tool utilizes SQL execution inside, it can solve complex subquery on table through one tool call.
4. Generate exactly ONE subquery at a time.
5. Write out all terms completely - avoid using abbreviations.
6. When you have sufficient information, provide the final answer in the following format:
    <Answer>: [your complete response]

User Input Query: {query}
Please start!
"""

COMBINE_PROMPT = """You are about to complete a table-based question answernig task using the following two types of reference materials:

# Content 1: Original content (table content is provided in Markdown format):
{docs}

$ Content 2: NL2SQL related Question information and SQL execution result in the database:
# the user given table schema
{schema}

# SQL generated based on the schema and the user question:
{nl2sql_model_response}

# SQL execution results
{sql_execute_result}

Please answer the user's question based on the materials above.
User question: {query}

Note:
1. The markdown table content in Content 1 may be incomplete.
2. You should cross-validate the given two materials:
    - if the answers are the same, directly output the answer.
    - if the "SQL execution result" contains error or is empty, you should try to answer based on the Content 1.
    - if the two materials shows conflit, you should think about each of them, and finally give an answer.
"""

EVALUATION_PRONPT = """We would like to request your feedback on the performance of the AI assistant in response to the user question displayed above according to the gold answer. Please use the following listed aspects and their descriptions as evaluation criteria:
    - Accuracy and Hallucinations: The assistant's answer is semantically consistent with the gold answer; The numerical value and order need to be accurate, and there should be no hallucinations.
    - Completeness: Referring to the reference answers, the assistant's answer should contain all the key points needed to answer the user's question; further elaboration on these key points can be omitted.
Please rate whether this answer is suitable for the question. Please note that the gold answer can be considered as a correct answer to the question.

The assistant receives an overall score on a scale of 0 OR 1, where 0 means wrong and 1 means correct.
Dirctly output a line indicating the score of the Assistant.

PLEASE OUTPUT WITH THE FOLLOWING FORMAT, WHERE THE SCORE IS 0 OR 1 BY STRICTLY FOLLOWING THIS FORMAT: "[[score]]", FOR EXAMPLE "Rating: [[1]]":
<start output>
Rating: [[score]]
<end output> 

[Question]
{question}

[Gold Answer]
{golden}

[The Start of Assistant's Predicted Answer]
{gen}
"""
