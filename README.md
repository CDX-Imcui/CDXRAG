# TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning

Repo for _[TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning](https://github.com/yxh-y/TableRAG/)_  

![Main Architecture](./figures/Main%20structure.png)

# 📌 Introduction

- We identify two key limitations of existing RAG approaches in the context of heterogeneous document question answering: structural information loss and lack of global view. 
- We propose **TableRAG**, an **Hybrid (SQL Execution and Textual Retrieval) framework** that unifies textual understanding and complex manipulations over tabular data. TableRAG comprises an offline database construction phase and a four-step online iterative reasoning process.
- We develop **HeteQA**, a benchmark for evaluating multi-hop heterogeneous reasoning capabilities. Experimental results show that TableRAG outperforms RAG and programmatic approaches on HeteQA and public benchmarks, establishing a state-of-the-art solution.

# 🔎 Setup

## Environment
```
conda create -n your_env python=3.10

git clone https://github.com/yxh-y/TableRAG/
cd TableRAG

pip install -r requirements.txt
```

# 🛠 How to Run?

## Dataset Preparation
1. Download dev_excel.zip from [Google Drive](https://drive.google.com/drive/folders/1Pea6kiUZv0UP8k7Ohv19KorBdBaUrouE?usp=drive_link).

## Offline Workflow

### Step 1: Setup MySQL Database

1. Download MySQL
Reach https://downloads.mysql.com/archives/community/ and find MySQL 8.0.24 and downloads for your appropriate environment.

2. Install MySQL
```
tar -zxvf mysql-8.0.24-linux-glibc2.12-x86_64.tar.gz
cd mysql-8.0.24-linux-glibc2.12-x86_64
sudo mkdir /usr/local/mysql && sudo mv * /usr/local/mysql/
sudo groupadd mysql
sudo useradd -r -g mysql mysql
cd /usr/local/mysql
sudo bin/mysqld --initialize --user=mysql --basedir=/usr/local/mysql --datadir=/usr/local/mysql/data
sudo cp support-files/mysql.server /etc/init.d/mysql
sudo systemctl enable mysql
sudo systemctl start mysql
```
3. Create Database for TableRAG
```sql
CREATE DATABASE TableRAG;
```

### Step 2: Offline data Ingestion

1. Setup database config 
Edit offline_data_ingestion_and_query_interface/config/database_config.json and update it with your own MySQL config.

2. Prepare table files to be ingested
Unzip dev_excel.zip to 'offline_data_ingestion_and_query_interface/dataset/hybridqa/dev_excel/'.

4. Execute data ingestion pipeline
```
cd offline_data_ingestion_and_query_interface/src/
python data_persistent.py
```

### Step 3: Start Database query service

1. Setup LLM config
Edit 'offline_data_ingestion_and_query_interface/src/handle_requests.py' and substitute your llm request url and apikey into model_request_config.

2. Start service to provide SQL query interface

```
python interface.py
```

## Online Workflow

### Step 1: Setup Config and Data Source

1. Edit 'online_inference/config.py' to set the LLM infering url and key, and the query service url.
   
3. Unzip the dev_excel.zip and put it into "/data" directory.

### Step 2: Run Main Experiment
```
cd online_inference
python3 main.py
  --backbone <backbone_llm>
  --data_file_path ./data/my_dev.json
  --save_file_path <path to save file>
  --max_iter <max iterations of TableRAG, default to 5>
  --rerun <True if some cases fail at the previous run, default to False> 
  
 python main.py   --backbone qwen-flash  --data_file_path ./data/my_dev.json  --doc_dir ./data/dev_doc  --excel_dir ./data/dev_excel --bge_dir ./models --save_file_path ./output.json
 python main.py   --backbone qwen2.5-72b-instruct  --data_file_path ./data/my_dev.json  --doc_dir ./data/dev_doc  --excel_dir ./data/dev_excel --bge_dir ./models --save_file_path ./output.json
```



