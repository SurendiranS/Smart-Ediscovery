# Smart-Ediscovery
Smart eDiscovery using Large Language Model with Streamlit

## Description
This project implemented using [Langchain framework](https://github.com/hwchase17/langchain) and [Chroma DB](https://github.com/chroma-core/chroma), an open-source embedding database. It uses [LaMini-T5-738M model](https://huggingface.co/MBZUAI/LaMini-T5-738M#lamini-t5-738m) a fine-tuned version of [t5-large](https://huggingface.co/t5-large) on [LaMini-instruction dataset](https://huggingface.co/datasets/MBZUAI/LaMini-instruction) which contains 2.58M samples for instruction fine-tuning.

## Running the Code
To install the requirements for this project, run the following command:
```bash
$ pip install -r requirements.txt
```
To use this project, run the [app.py](https://github.com/SurendiranS/Smart-Ediscovery/blob/main/app.py) script 
```bash
$ streamlit run app.py
```
![image](https://github.com/SurendiranS/Smart-Ediscovery/assets/43315429/0f4de6da-ed44-4321-9cd2-831f72126c2c)
