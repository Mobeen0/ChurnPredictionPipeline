.PHONY: run

run:
	uvicorn Module4_api:app --reload & streamlit run Module4_frontend.py
