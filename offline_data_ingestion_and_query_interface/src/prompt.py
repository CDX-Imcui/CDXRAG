

NL2SQL_SYSTEM_PROMPT = "You are an expert in SQL and can generate SQL statements based on table schemas and query requirements. Respond as concisely as possible, providing only the SQL statement without any additional explanations."

NL2SQL_USER_PROMPT = '''{schema_list}
Based on the schemas above, please use MySQL syntax to solve the following problem:
{user_query}
Please wrap the generated SQL statement with ```sql ```, and warp table name and each column name metioned in sql with ``, for example: ```sql SELECT `name` FROM `student_sheet1` WHERE `age` > '15';```
'''