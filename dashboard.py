import streamlit as st
import pm4py
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

# --- CONFIGURATION ---
st.set_page_config(page_title="Process Mining Dashboard", layout="wide")
load_dotenv(override=True)
FILENAME = "BPI_Challenge_2013_closed_problems.xes.gz"
API_KEY = os.getenv('OPENAI_API_KEY')

# --- DATA LOADING ---
@st.cache_data
def load_data():
    log = pm4py.read_xes(FILENAME)
    df = pm4py.convert_to_dataframe(log)
    
    # Rename columns to avoid Altair/Streamlit errors
    rename_map = {
        'concept:name': 'Activity',
        'org:resource': 'Resource',
        'case:concept:name': 'Case_ID',
        'time:timestamp': 'Timestamp'
    }
    cols_to_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=cols_to_rename)
    
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
        
    return log, df

@st.cache_data
def get_process_stats(df):
    if 'Case_ID' in df.columns:
        num_cases = len(df['Case_ID'].unique())
    else:
        num_cases = 0 
    num_events = len(df)
    return num_cases, num_events

def ask_openai(prompt, system_role="You are a Process Mining Expert."):
    if not API_KEY:
        return "⚠️ ERROR: API Key missing."
    
    client = OpenAI(api_key=API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API ERROR: {e}"

# --- MAIN APP ---

st.title("📊 Process Mining Dashboard - BPI 2013")

if os.path.exists(FILENAME):
    log, df = load_data()
    cases, events = get_process_stats(df)
else:
    st.error(f"File {FILENAME} not found!")
    st.stop()

# Sidebar
st.sidebar.header("Dataset Info")
st.sidebar.metric("Total Cases", cases)
st.sidebar.metric("Total Events", events)
st.sidebar.markdown("---")
st.sidebar.info("Dataset: BPI Challenge 2013 (Closed Problems)")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📂 Data View", "🔍 Process Discovery", "💬 Chat with AI", "🔮 Prediction with AI"])

# --- TAB 1: DATA VIEW ---
with tab1:
    st.header("Data Exploration")
    st.dataframe(df.head(100), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Activities")
        if 'Activity' in df.columns:
            st.bar_chart(df['Activity'].value_counts())
    with col2:
        st.subheader("Top Resources")
        if 'Resource' in df.columns:
            st.bar_chart(df['Resource'].value_counts().head(10))
    
    # --- ANOMALY DETECTION SECTION ---
    st.divider()
    st.subheader("Anomaly Detection")
    
    col_anom1, col_anom2 = st.columns(2)
    
    with col_anom1:
        st.markdown("**📊 Resource Workload Analysis**")
        st.caption("Analyzes how events are distributed among resources to identify bottlenecks")
        if 'Resource' in df.columns:
            resource_counts = df['Resource'].value_counts()
            mean_workload = resource_counts.mean()
            std_workload = resource_counts.std()
            num_resources = len(resource_counts)
            
            # Identify overloaded resources (>2 std above mean)
            overloaded = resource_counts[resource_counts > mean_workload + 2 * std_workload]
            
            if len(overloaded) > 0:
                st.warning(f"⚠️ **Overloaded Resources** (>2σ above mean): {', '.join(overloaded.index.tolist())}")
            else:
                st.success("✅ No severely overloaded resources detected")
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Avg Events/Resource", f"{mean_workload:.1f}", 
                         help="Average number of events handled by each resource. Lower = more balanced workload")
            with col_m2:
                st.metric("Workload Std Dev", f"{std_workload:.1f}",
                         help="Variability in workload. High value = uneven distribution (some resources handle much more)")
            
            st.caption(f"📌 Total resources: **{num_resources}** | Workload range: **{resource_counts.min()}** - **{resource_counts.max()}** events")
    
    with col_anom2:
        st.markdown("**🔄 Process Complexity Indicators**")
        st.caption("Detects cases with unusual patterns that may indicate process issues")
        if 'Case_ID' in df.columns and 'Activity' in df.columns:
            # Calculate events per case
            events_per_case = df.groupby('Case_ID').size()
            avg_events = events_per_case.mean()
            max_events = events_per_case.max()
            min_events = events_per_case.min()
            
            # Detect cases with unusually many events (potential loops)
            complex_cases = events_per_case[events_per_case > avg_events + 2 * events_per_case.std()]
            
            col_m3, col_m4 = st.columns(2)
            with col_m3:
                st.metric("Avg Events/Case", f"{avg_events:.2f}",
                         help="Average number of activities per incident. Higher = more complex process")
            with col_m4:
                st.metric("Max Events in Case", max_events,
                         help="Highest number of events in a single case. Very high = potential loops or rework")
            
            st.caption(f"📌 Events/case range: **{min_events}** - **{max_events}** | Normal range: **{avg_events - events_per_case.std():.1f}** - **{avg_events + events_per_case.std():.1f}**")
            
            if len(complex_cases) > 0:
                st.warning(f"⚠️ **{len(complex_cases)} cases** ({len(complex_cases)/len(events_per_case)*100:.1f}%) with unusually high event count (potential loops)")
            else:
                st.success("✅ No excessively complex cases detected")

# --- TAB 2: DISCOVERY ---
with tab2:
    st.header("Model Comparison")
    selected_model = st.radio("Choose algorithm:", ["Alpha Miner", "Heuristic Miner", "Inductive Miner"], horizontal=True)
    
    col_img, col_metrics = st.columns([2, 1])
    with col_img:
        st.subheader("Petri Net")
        img_path = f"model_{selected_model.lower().replace(' ', '_')}.png"
        if os.path.exists(img_path):
            st.image(img_path, caption=f"Petri Net generated using {selected_model}", use_container_width=True)
        else:
            st.warning("Image not found. Run discovery.py.")
            
    with col_metrics:
        st.subheader("Metrics")
        metrics_path = f"metrics_{selected_model.lower().replace(' ', '_')}.png"
        if os.path.exists(metrics_path):
            st.image(metrics_path, use_container_width=True)

    if selected_model == "Inductive Miner":
            st.divider()
            st.success("✅ **BEST MODEL**: because has the ability to guarantee soundness (formal correctness) and perfect fitness (1.0).")

# --- TAB 3: CHATBOT (Optimized) ---
with tab3:
    st.header("Chat with AI")
    
    # 0. Static Report (in an expander to save space)
    with st.expander("📄 View Generated Business Report (Click to expand)"):
        if os.path.exists("ai_report.md"):
            with open("ai_report.md", "r") as f:
                st.markdown(f.read())
        else:
            st.info("Run `reasoning.py` to generate the initial report.")

    st.divider()
    # 1. Message container (allows native scrolling)
    #    In Streamlit, st.chat_message automatically handles WhatsApp-style UI
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Optional welcome message
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I analyzed the Incident Management process. Ask me anything about bottlenecks, resources, or timing."})

    # Display history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.markdown("""
    <style>
        .stChatInput {
            position: fixed;
            bottom: 50px;
            left: -9;
            width: 70%;
            margin-right: 5vw;
            padding: 10px;
            z-index: 20;
        }
    </style>
    """, unsafe_allow_html=True)

    # 2. Fixed Input at Bottom
    if prompt := st.chat_input("Ask a question about the process (e.g., *“What is the main bottleneck?”* or *“How can resources be optimized?”*)"):
        # Add and display user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Response logic
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Enhanced dynamic context with anomaly info
                top_acts = df['Activity'].value_counts().head(5).to_dict() if 'Activity' in df.columns else "N/A"
                top_resources = df['Resource'].value_counts().head(5).to_dict() if 'Resource' in df.columns else "N/A"
                
                # Calculate anomaly indicators
                if 'Case_ID' in df.columns:
                    events_per_case = df.groupby('Case_ID').size()
                    avg_events_case = events_per_case.mean()
                    max_events_case = events_per_case.max()
                else:
                    avg_events_case = max_events_case = "N/A"
                
                avg_events_case_str = (
                    f"{avg_events_case:.2f}"
                    if isinstance(avg_events_case, float)
                    else str(avg_events_case)
                )
                
                context = f"""Dataset: BPI Challenge 2013 (Volvo IT Incident Management).
                Total Cases: {cases}. Total Events: {events}.
                Activities: {top_acts}.
                Top Resources: {top_resources}.
                Avg Events/Case: {avg_events_case_str}. Max Events in a Case: {max_events_case}.
                Note: Look for anomalies like resource bottlenecks, excessive loops, or unusual patterns."""
                
                full_prompt = f"Context: {context}\nUser Question: {prompt}\nAnswer as a concise Business Analyst. If the question is about anomalies, highlight specific issues found in the data."
                response = ask_openai(full_prompt)
                
                st.markdown(response)
        
        # Save response in history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Helper function to calculate what usually happens after an activity
def get_next_activity_stats(df, last_activity):
    # Find all rows where our activity is present
    # Note: This is simplified. For absolute precision we should iterate traces.
    # But for the dashboard, looking at sequential pairs is fine.
    
    # Create pairs (Current Activity, Next Activity)
    next_activities = df['Activity'].shift(-1)
    mask = (df['Activity'] == last_activity) & (df['Case_ID'] == df['Case_ID'].shift(-1))
    
    # Count occurrences of following activities
    counts = next_activities[mask].value_counts(normalize=True).head(3) # Take top 3
    
    if counts.empty:
        return "No historical data available for this step."
    
    # Format into readable string
    stats_str = ", ".join([f"{act} ({perc:.1%})" for act, perc in counts.items()])
    return stats_str

# --- TAB 4: PREDICTION LAB (Advanced) ---
with tab4:
    st.header("Prediction with AI")
    st.markdown("""
    Prediction of events with AI using **In-Context Learning**.\n
    We feed the AI not just the "history", but also the **statistical rules** derived from the dataset.
    """)
    st.divider()

    # 1. Select Case
    all_cases = df['Case_ID'].unique()
    selected_case = st.selectbox("Select a Case ID to analyze:", all_cases[:100])

    if selected_case:
        # Extract case data
        case_data = df[df['Case_ID'] == selected_case].sort_values('Timestamp').reset_index(drop=True)
        full_trace = case_data['Activity'].tolist()
        # Involved resources
        resources_trace = case_data['Resource'].tolist()
        total_steps = len(full_trace)

        # Show data
        with st.expander("📄 View Case Log", expanded=False):
            st.dataframe(case_data[['Activity', 'Resource', 'Timestamp']], use_container_width=True)

        if total_steps > 2:
            st.subheader("⏳ Simulation Control")
            cut_point = st.slider(
                "Stop timeline at event number:", 
                min_value=1, 
                max_value=total_steps-1, 
                value=total_steps-1
            )

            # Data at cutoff point
            past_events = full_trace[:cut_point]
            last_event = past_events[-1]
            last_resource = resources_trace[cut_point-1]
            
            real_next_event = full_trace[cut_point]
            
            # --- CALCULATE KNOWLEDGE (STATS) ---
            # Here happens the magic: calculate what usually happens in the dataset
            stats_knowledge = get_next_activity_stats(df, last_event)

            # Context Visualization
            st.write(f"**Scenario:**")
            st.info(f"📜 History: {' -> '.join(past_events)}")
            st.caption(f"👤 Last Resource: **{last_resource}** | 📊 Dataset Stats: After '{last_event}', usually comes: **{stats_knowledge}**")

            if st.button("🚀 Predict with Knowledge", type="primary"):
                col_ai, col_real = st.columns(2)
                
                with col_ai:
                    st.markdown("### 🔮 AI Prediction")
                    with st.spinner("Consulting knowledge base..."):
                        
                        # --- ENRICHED PROMPT ---
                        pred_prompt = f"""
                        Task: Predict the next activity in an Incident Management process (Volvo IT).
                        
                        --- CONTEXT DATA ---
                        1. Trace History: {past_events}
                        2. Last Activity: "{last_event}"
                        3. Last Resource: "{last_resource}"
                        Possible activities: Queued, Accepted, Completed, Unmatched.
                        
                        --- KNOWLEDGE BASE (Historical Stats from Dataset) ---
                        In this specific dataset, after "{last_event}", the following usually happens:
                        {stats_knowledge}
                        
                        --- INSTRUCTIONS ---
                        Based on the History AND the Knowledge Base stats, what is the SINGLE most likely NEXT activity?
                        If the stats are strong (e.g. >50%), trust them. If ambiguous, use logical workflow inference.
                        
                        Return ONLY the activity name! ONLY! No other things!!!!!!!!!!!
                        """
                        
                        prediction = ask_openai(pred_prompt, system_role="You are a Data-Driven Process Predictor.")
                        st.warning(f"**{prediction}**")

                with col_real:
                    st.markdown("### 👁️ Ground Truth")
                    st.success(f"**{real_next_event}**")
                
                # Verification
                if prediction.strip().lower() in real_next_event.lower():
                    st.success("✅ AI guessed correctly!")
                else:
                    st.error("❌ Divergence (AI followed logic/stats, but reality was different)")
                    
        else:
            st.warning("Case too short for prediction.")