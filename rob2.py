import streamlit as st
import ollama
import json
import pandas as pd
import io
import os
from PyPDF2 import PdfReader
from openai import OpenAI

# --- Environment Detection ---
def is_running_on_cloud():
    return os.environ.get("STREAMLIT_RUNTIME_ENV") is not None or os.environ.get("HOSTNAME") == "streamlit"

IS_CLOUD = is_running_on_cloud()

# --- Page Configuration ---
st.set_page_config(page_title="RoB-2 Automated Ratings", page_icon="💊", layout="wide")

# --- Function to Fetch Local Models ---
def get_local_models():
    if IS_CLOUD: return []
    try:
        client = ollama.Client()
        models_info = client.list()
        if 'models' in models_info:
            return sorted([m.get('name') or m.get('model') for m in models_info['models']])
        return []
    except:
        return []

# --- UI Header ---
st.title("💊 RoB-2 Automated Ratings")
st.markdown("Automated Risk of Bias assessment using AI (Local LLMs or Cloud APIs).")

# --- Instructions (Didactic Interface) ---
with st.expander("📖 How to use", expanded=True):
    st.markdown("""
    ### Prerequisites
    1. **Cloud Mode (API):** Get a free API key at the [Groq Console](https://console.groq.com/keys) (fastest/easiest) or use OpenRouter as an alternative.
    2. **Local Mode (Ollama):** Ensure Ollama is running. For accurate results, use models like `deepseek-r1:32b` or higher. Use `gemma2:9b` only for quick functionality tests.
    
    ### Steps
    1. **Upload:** Drag and drop clinical trial PDFs (RCTs).
    2. **Processing:** The system extracts text, sends it to the selected LLM, and parses the 5 RoB-2 domains.
    3. **Retry Logic:** If the AI fails to format the JSON or misses data, the system will automatically retry up to 3 times.
    """)

# --- Sidebar Settings (Updated with "Other" option for Cloud) ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    mode_options = ["Cloud API", "Local Ollama"]
    selected_mode = st.radio("Processing Mode", mode_options, index=0 if IS_CLOUD else 1)

    if selected_mode == "Local Ollama" and IS_CLOUD:
        st.error("❌ Local Ollama is unavailable on the web version.")
        st.stop()

    st.divider()

    if selected_mode == "Cloud API":
        provider = st.selectbox("Provider", ["Groq", "OpenRouter", "OpenAI"])
        
        # --- Secrets Logic ---
        secret_key_name = f"{provider.upper()}_API_KEY"
        secret_key = st.secrets.get(secret_key_name)
        
        if secret_key:
            api_key = secret_key
            st.info(f"✅ {provider} API Key loaded from system.")
        else:
            api_key = st.text_input(f"{provider} API Key", type="password", help="Enter your key manually.")
        
        # Updated models list for Cloud
        if provider == "OpenAI":
            cloud_options = ["gpt-5-chat-latest", "o3-2025-04-16", "Other (Type name...)"]
        elif provider == "Groq":
            cloud_options = ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b", "llama3-70b-8192", "Other (Type name...)"]
        else:
            cloud_options = ["deepseek/deepseek-r1", "llama-3.3-70b-versatile", "Other (Type name...)"]
        selected_cloud_model = st.selectbox("Model", cloud_options)
        model_name = st.text_input("Enter model name (e.g., gpt-oss-120b):") if "Other" in selected_cloud_model else selected_cloud_model.split(" ")[0]
            
    else:
        # Local Ollama Mode
        if st.button("🔄 Refresh Models"): st.rerun()
        actual_models = get_local_models()
        local_options = actual_models + ["Other (Type name...)"]
        selected_local_option = st.selectbox("Select Local Model", options=local_options)
        
        if selected_local_option == "Other (Type name...)":
            model_name = st.text_input("Enter Local Model Name (exactly as in Ollama):")
        else:
            model_name = selected_local_option
    
    # FORA do else, mas ainda DENTRO do 'with st.sidebar'
    st.divider()
    target_outcome = st.text_input(
        "Target Outcome", 
        value="Primary Outcome", 
        help="Define which outcome the AI should focus on (e.g., All-cause mortality, Adverse Events)."
    )

    output_format = st.selectbox("Export Format", [".csv", ".xlsx"])
    
    st.divider()
    st.caption("Contact: pedro.vidor@ufrgs.br")
    
# --- Unified Inference Function ---
def call_llm(prompt_content):
    if selected_mode == "Cloud API":
        if not api_key:
            st.error("Please provide an API Key.")
            return None
        
        if provider == "Groq":
            base_url = "https://api.groq.com/openai/v1"
        elif provider == "OpenRouter":
            base_url = "https://openrouter.ai/api/v1"
        else:
            base_url = None # Padrão da biblioteca OpenAI
            
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_content}],
            response_format={ "type": "json_object" }
        )
        return response.choices[0].message.content
    else:
        # Local Ollama logic
        response = ollama.generate(model=model_name, prompt=prompt_content, format="json")
        return response['response']

# --- "Lai et al. 2024" Prompt ---
EXPERT_SYSTEM_PROMPT = """
Introduction and Role Setting:

You are a professional reviewer. You are particularly good at learning evaluation criteria, and closely following it to assess the risk of bias of Randomized Controlled Trials (RCTs). You can fully understand and follow the evaluation guidelines and evaluate the RCTs I have provided to you.

Make sure all your judgments are based on the facts reported in the article and not on any extrapolation or speculation of your own. Finally, make sure your answers are completely correct.

Guidelines for Evaluation:

Note: The examples provided in the tool are illustrative and do not cover all possible scenarios in real-world applications. Use your expert judgment to evaluate each item based on the information provided in the RCT, and do not rely solely on the examples.

Important:


The evaluation should be conducted only for one selected outcome.

If there is too little information to support the judgment, do not speculate positively.

Evaluation Items:

1. Was the allocation sequence adequately generated?

Evaluate the adequacy of the allocation sequence generation based on the information provided in the RCT, considering the following criteria:


The most important: If no statements are provided on how the randomization sequence was generated, select "Probably no", even if randomization is mentioned.

If computer-generated random numbers, coin tossing, card or envelope shuffling, dice rolling, lot drawing, or minimization (with or without a random element) were used, select “Definitely yes.”

If the sequence was generated based on the odd or even date of birth, some rule based on the date (or day) of admission, or some rule based on hospital or clinic record number, carefully evaluate and choose between “Probably yes” and “Probably no.”

If allocation was based on clinician judgment, participant preference, results of a series of laboratory tests, or availability of the intervention, select “Definitely no.”

2. Was the allocation adequately concealed?

Evaluate the adequacy of allocation concealment based on the information provided in the RCT, considering the following criteria:


If central allocation (including telephone, web-based, and pharmacy-controlled randomization), sequentially numbered drug containers of identical appearance, or sequentially numbered, opaque, sealed envelopes were used, select “Definitely yes.”

If an open random allocation schedule was used or if assignment envelopes were used without appropriate safeguards (e.g., if envelopes were unsealed, non-opaque, or not sequentially numbered), select “Definitely no.”

If no statements are provided on allocation concealment, select “Probably no.”

3. Blinding: Was knowledge of the allocated interventions adequately prevented?

Evaluate the adequacy of blinding for each of the following, based on the information provided in the RCT:


3.a. Were patients blinded?

3.b. Were healthcare providers blinded?

3.c. Were data collectors blinded?

3.d. Were outcome assessors blinded?

3.e. Were data analysts blinded?

Standards for 3.a. to 3.e.:


If no blinding but you judge that the outcome and the outcome measurement are not likely influenced by lack of blinding, select “Probably yes.”

If blinding of participants and key study personnel ensured, and unlikely that blinding could have been broken, select “Probably yes.”

If either participants or some key study personnel were not blinded, but outcome assessment was blinded and the nonblinding of others unlikely to introduce bias, select “Probably yes.”

If no blinding or incomplete blinding, and the outcome or outcome measurement is likely to be influenced by lack of blinding, select “Probably no.”

If blinding of key study participants and personnel attempted, but likely that the blinding could have been broken, select “Probably no.”

If either participants or some key study personnel were not blinded, and the nonblinding of others likely to introduce bias, select “Probably no.”

4. Was loss to follow-up (missing outcome data) infrequent?

Evaluate the frequency of loss to follow-up based on the information provided in the RCT, considering the following criteria:


If there are no missing outcome data, or the reasons for missing outcome data are unlikely to be related to the outcome, select “Definitely yes.”

If missing outcome data are balanced across intervention groups, with similar reasons for missing data across groups, select “Probably yes.”

If the proportion of missing outcomes is enough to have an important impact on the intervention effect estimate, select “Definitely no.”

If the follow-up rate at the longest time point is greater than 90%, i.e. more than 90% of participants completed the trial, or the dropout rate at the longest time point is less than 10%, you can generally consider selecting “Definitely yes.”

If the follow-up rate is between 80% and 90%, i.e. more than 80% but less than 90% of participants completed the trial, or the dropout rate at the longest time point is between 10% and 20%, consider selecting “Probably yes.”

If the follow-up rate is below 80%, i.e. less than 80% of participants completed the trial, or the dropout rate is greater than 20%, consider selecting “Definitely no.”

If no statements are available to support the assessment, select “Probably no.”

5. Are reports of the study free of selective outcome reporting?

Evaluate the presence of selective outcome reporting based on the information provided in the RCT, considering the following criteria:


If the study protocol is available and all of the study’s pre-specified (primary and secondary) outcomes of interest in the review have been reported in the pre-specified way, select “Definitely yes.”

If the study protocol is not available but it is clear that the published reports include all expected outcomes, including those that were pre-specified, select “Probably yes.”

If not all of the study’s pre-specified primary outcomes have been reported, or if one or more reported primary outcomes were not pre-specified, select “Definitely no.”

Don't select “Probably no” just because the study protocol is not available

6. Was the study apparently free of other problems that could put it at a risk of bias?

Evaluate the presence of other potential sources of bias based on the information provided in the RCT, considering the following criteria:


If the study appears to be free of other sources of bias, select “Definitely yes.”

If the study had a potential source of bias related to the specific study design used, or had some other problem that could put it at risk of bias, select “Probably no” or “Definitely no.”

Output Format:


- You must output ONLY a JSON object.

- Make all judgments based on facts. If information is missing, select "Probably no".


JSON Structure:

{

  "D1": {"judgment": "definitely yes / probably yes / probably no / definitely no", "support": "Reasoning + excerpt with <<quote>> -- be sure to use only '<<' and '>>' before and after the excerpt to avoid document errors"},
  "D2": {"judgment": "definitely yes / probably yes / probably no / definitely no", "support": "Reasoning + excerpt with <<quote>> -- be sure to use only '<<' and '>>'"},
  "D3": {"judgment": "definitely yes / probably yes / probably no / definitely no", "support": "Reasoning + excerpt with <<quote>> -- be sure to use only '<<' and '>>'"},
  "D4": {"judgment": "definitely yes / probably yes / probably no / definitely no", "support": "Reasoning + excerpt with <<quote>> -- be sure to use only '<<' and '>>'"},
  "D5": {"judgment": "definitely yes / probably yes / probably no / definitely no", "support": "Reasoning + excerpt with <<quote>> -- be sure to use only '<<' and '>>'"},
  "Overall": {"judgment": "definitely yes / probably yes / probably no / definitely no", "support": "Final summary based on the domains above"}

}




Ensure there is a clear separation between the sets of responses for different outcomes, and maintain consistency in the format for each outcome.

Before you output the answer, please make sure that any evaluation results are based on my request.

Please re-check: In the first item, if no statements are available on how the randomization sequence was generated, select "Probably no", even if randomization is mentioned. In each item, if no statements are available to support the assessment, select “Probably no.” But be careful




"""

def get_pdf_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content: text += content + " "
    return " ".join(text.split())

def safe_map_judgment(res_json, domain):
    data = res_json.get(domain, {})
    if not isinstance(data, dict): return "N/A"
    j = data.get('judgment', 'N/A').lower()
    if any(x in j for x in ["definitely yes", "probably yes", "low"]): return "Low"
    if any(x in j for x in ["definitely no", "probably no", "high"]): return "High"
    if "some" in j: return "Some concerns"
    return "N/A"

# --- Main Processing ---
uploaded_files = st.file_uploader("Upload Study PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files and st.button("Start Batch Processing"):
    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for index, file in enumerate(uploaded_files):
        raw_text = get_pdf_text(file)
        study_context = raw_text[:16000] 
        
        success = False
        attempts = 0
        while attempts < 3 and not success:
            attempts += 1
            status_text.text(f"Analyzing {file.name} (Attempt {attempts}/3)...")
            
            try:
                # Dentro do loop 'for file in uploaded_files:'
                full_prompt = f"TARGET OUTCOME TO EVALUATE: {target_outcome}\n\n{EXPERT_SYSTEM_PROMPT}\n\nArticle Content:\n{study_context}"
                response_text = call_llm(full_prompt)
                res_json = json.loads(response_text)
                
                temp_res = {
                    "Study": file.name.replace(".pdf", ""),
                    "D1": safe_map_judgment(res_json, 'D1'),
                    "D2": safe_map_judgment(res_json, 'D2'),
                    "D3": safe_map_judgment(res_json, 'D3'),
                    "D4": safe_map_judgment(res_json, 'D4'),
                    "D5": safe_map_judgment(res_json, 'D5'),
                    "Overall": safe_map_judgment(res_json, 'Overall'),
                    "Full_JSON": res_json
                }
                
                # Simple validation to ensure JSON is complete
                if not any(temp_res[k] == "N/A" for k in ["D1", "D2", "D3", "D4", "D5"]):
                    all_results.append(temp_res)
                    success = True
            except Exception as e:
                if attempts == 3: st.error(f"Error in {file.name}: {e}")

        progress_bar.progress((index + 1) / len(uploaded_files))

    if all_results:
        df = pd.DataFrame(all_results)
        view_cols = ["Study", "D1", "D2", "D3", "D4", "D5", "Overall"]
        
        st.subheader("Analysis Results")
        st.dataframe(df[view_cols])

        # --- Downloads ---
        c1, c2 = st.columns(2)
        with c1:
            if output_format == ".csv":
                st.download_button("📥 Download CSV", df[view_cols].to_csv(index=False).encode('utf-8'), "rob2_results.csv")
            else:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                    df[view_cols].to_excel(writer, index=False)
                st.download_button("📥 Download Excel", buf.getvalue(), "rob2_results.xlsx")
        
        with c2:
            full_csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📑 Download Full Audit Report", full_csv, "rob2_full_audit.csv")

        with st.expander("🔍 View AI Justifications"):
            for res in all_results:
                st.write(f"### Study: {res['Study']}")
                st.json(res['Full_JSON'])

st.divider()

st.caption("2026 | Integration with local AIs is available only in the desktop version.")

