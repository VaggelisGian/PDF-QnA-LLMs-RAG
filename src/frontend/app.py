import streamlit as st
import requests
import time
import os
import json
from datetime import datetime
import logging
from typing import Optional

try:
    for name, logger_instance in logging.root.manager.loggerDict.items():
        if "streamlit" in name:
            logger_instance.disabled = True
    if "streamlit" in logging.root.manager.loggerDict:
         logging.getLogger("streamlit").disabled = True
except Exception as e:
    print(f"Error disabling loggers: {e}")


BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
print(f"Using backend URL: {BACKEND_URL}")


def check_services_ready():
    try:
        health_url = f"{BACKEND_URL}/api/health"
        print(f"DEBUG: Checking backend health at {health_url}")
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            print("DEBUG: Backend health check successful (status 200).")
            return True
        else:
            print(f"DEBUG: Backend health check failed with status {response.status_code}.")
            return False
    except requests.exceptions.ConnectionError:
        print("DEBUG: Backend health check failed (Connection Error).")
        return False
    except Exception as e:
        print(f"DEBUG: Backend health check failed with exception: {e}")
        return False

def update_debug_mode():
    st.session_state.debug_mode = st.session_state.debug_checkbox_key

def call_chat_api(question: str, use_graph: bool = False, temperature: Optional[float] = None, max_tokens: Optional[int] = None):
    try:
        chat_url = f"{BACKEND_URL}/api/chat"
        payload = {"question": question, "use_graph": use_graph}

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        print(f"DEBUG: Calling chat API with payload: {payload}")
        response = requests.post(chat_url, json=payload, timeout=400)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Chat API error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None


def poll_progress(job_id):
    try:
        progress_url = f"{BACKEND_URL}/api/progress/{job_id}"

        response = requests.get(progress_url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            st.session_state.progress_data = data
            st.session_state.last_poll_time = time.time()

            status = data.get("status", "processing")


            if status in ["completed", "failed", "error", "completed_empty"]:
                print(f"DEBUG: Job {job_id} status is {status}. Stopping processing.")
                st.session_state.processing_active = False
                return False
            else:
                return True
        elif response.status_code == 404:
             print(f"DEBUG: Poll failed with 404 for job {job_id}. Assuming job finished or invalid. Stopping polling.")
             st.session_state.processing_active = False

             st.session_state.progress_data = {
                 **(st.session_state.progress_data or {}),
                 "job_id": job_id,
                 "status": "error",
                 "message": f"Polling failed: Job ID {job_id} not found (404)."
             }
             return False
        else:
            print(f"DEBUG: Poll failed with status {response.status_code} - {response.text[:200]}")

            st.session_state.last_poll_time = time.time()

            return True

    except requests.exceptions.Timeout:
         print(f"DEBUG: Timeout during polling job {job_id}.")
         st.session_state.last_poll_time = time.time()

         return True
    except requests.exceptions.RequestException as e:
        print(f"DEBUG: Error during polling: {e}")
        st.session_state.last_poll_time = time.time()

        return True
    except Exception as e:
        print(f"DEBUG: Unexpected error during polling: {e}")
        st.session_state.last_poll_time = time.time()
        return True


def main():

    wait_placeholder = st.empty()
    while not check_services_ready():
        wait_placeholder.warning("⏳ Waiting for backend services (Neo4j, Redis, Backend) to start... This might take a minute.")
        print("DEBUG: Backend not ready, sleeping for 5 seconds...")
        time.sleep(5)
    wait_placeholder.empty()
    print("DEBUG: Backend services ready. Proceeding with app.")


    for key, default in [
        ("processing_active", False),
        ("current_job_id", None),
        ("uploaded_file_info", None),
        ("debug_mode", False),
        ("progress_data", None),
        ("chat_history", []),
        ("user_question", ""),
        ("last_poll_time", 0),
        ("llm_temp", 0.1),
        ("llm_max_tokens", 512),
        ("use_graph_rag", False),
        ("skip_graph_build", False)
    ]:
        if key not in st.session_state:
            st.session_state[key] = default


    needs_rerun_for_poll = False
    if st.session_state.processing_active and st.session_state.current_job_id:
        current_time = time.time()

        if current_time - st.session_state.last_poll_time >= 2.0:
            print(f"DEBUG: Time to poll job {st.session_state.current_job_id}")
            if poll_progress(st.session_state.current_job_id):

                 needs_rerun_for_poll = True
            else:

                 needs_rerun_for_poll = False
                 print("DEBUG: Polling indicated job finished or error. Rerunning once for final UI update.")
                 st.rerun()
        else:

             needs_rerun_for_poll = True

    st.title("Intelligent PDF Retriever!")
    st.sidebar.info(f"Connected to: {BACKEND_URL}")
    if st.sidebar.button("Refresh Connection"):

        if st.session_state.current_job_id and st.session_state.processing_active:
             print("DEBUG: Manual refresh button clicked while processing.")
             poll_progress(st.session_state.current_job_id)
             st.rerun()
        elif check_services_ready():
            st.sidebar.success("✅ Connection OK")
            st.rerun()
        else:
            st.sidebar.error("❌ Connection failed")
            st.rerun()

    st.header("1. Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file",
        type=["pdf", "txt"],
        key="doc_uploader",
        disabled=st.session_state.processing_active
    )


    if uploaded_file is not None and not st.session_state.processing_active:

        if st.session_state.uploaded_file_info is None or st.session_state.uploaded_file_info["name"] != uploaded_file.name:
            print(f"DEBUG: New file selected: {uploaded_file.name}")
            st.session_state.uploaded_file_info = {
                "name": uploaded_file.name,
                "bytes": uploaded_file.getvalue()
            }

            st.session_state.current_job_id = None
            st.session_state.progress_data = None
            st.session_state.processing_active = False
            st.session_state.chat_history = []
            st.session_state.last_poll_time = 0
            st.rerun()

    if st.session_state.uploaded_file_info:
        st.write("File selected:", st.session_state.uploaded_file_info["name"])


        st.session_state.skip_graph_build = st.checkbox(
            "Skip Graph Build (Faster, Disables Graph RAG & Topic Extraction)",
            value=st.session_state.skip_graph_build,
            key="skip_graph_build_checkbox",
            disabled=st.session_state.processing_active,
            help="Check this to only perform basic chunking and embedding for standard RAG. Topic extraction and entity relationship storage will be skipped."
        )


        st.checkbox(
            "Debug mode",
            value=st.session_state.debug_mode,
            key="debug_checkbox_key",
            on_change=update_debug_mode,
            disabled=st.session_state.processing_active
        )

        process_button_disabled = st.session_state.processing_active or st.session_state.uploaded_file_info is None
        if st.button("Process Document", disabled=process_button_disabled):
            print("DEBUG: Process Document button clicked.")
            st.session_state.processing_active = True
            st.session_state.current_job_id = None
            st.session_state.progress_data = None
            st.session_state.chat_history = []
            st.session_state.last_poll_time = 0
            st.rerun()


    if st.session_state.processing_active and st.session_state.current_job_id is None and st.session_state.uploaded_file_info:
        print("DEBUG: Entering upload logic block.")

        files = {"file": (st.session_state.uploaded_file_info["name"], st.session_state.uploaded_file_info["bytes"], "application/pdf")}
        form_data = {"skip_graph_build": str(st.session_state.skip_graph_build).lower()}


        upload_container = st.empty()
        with upload_container, st.status("Uploading file...", expanded=True) as status:
            try:
                upload_url = f"{BACKEND_URL}/api/upload"

                print(f"DEBUG: Posting file to {upload_url} with form data: {form_data}")
                response = requests.post(upload_url, files=files, data=form_data, timeout=60)


                if response.status_code == 200:
                    resp_json = response.json()
                    job_id = resp_json.get("job_id")
                    print(f"DEBUG: Upload successful, job ID: {job_id}")
                    st.session_state.current_job_id = job_id

                    st.session_state.progress_data = {
                        "job_id": job_id, "status": "starting",
                        "message": "Upload successful. Starting processing...",
                        "percent_complete": 0
                    }
                    st.session_state.last_poll_time = time.time()
                    st.session_state.processing_active = True
                    status.update(label="✅ Upload successful", state="complete", expanded=False)
                    print("DEBUG: Triggering rerun to start polling.")
                    st.rerun()
                else:

                    fail_msg = f"Upload failed: {response.status_code} - {response.text[:500]}"
                    print(f"DEBUG: {fail_msg}")
                    status.update(label=f"❌ {fail_msg}", state="error")
                    st.session_state.processing_active = False
                    st.session_state.current_job_id = None
                    st.rerun()
            except requests.exceptions.Timeout:

                fail_msg = "Error during upload: Request timed out."
                print(f"DEBUG: {fail_msg}")
                status.update(label=f"❌ {fail_msg}", state="error")
                st.session_state.processing_active = False
                st.session_state.current_job_id = None
                st.rerun()
            except Exception as e:

                fail_msg = f"Error during upload: {str(e)}"
                print(f"DEBUG: {fail_msg}")
                status.update(label=f"❌ {fail_msg}", state="error")
                st.session_state.processing_active = False
                st.session_state.current_job_id = None
                st.rerun()


    if st.session_state.current_job_id and st.session_state.progress_data:

        data = st.session_state.progress_data
        percent = data.get("percent_complete", 0)
        message = data.get("message", "Processing...")
        status_value = data.get("status", "processing")

        print(f"DEBUG RENDERING: Job={st.session_state.current_job_id}, Percent={percent}, Status={status_value}, Message={message}")

        progress_container = st.container()
        with progress_container:
            if status_value not in ["completed", "failed", "error", "completed_empty"]:
                st.info(f"Status: {status_value} - {message}")
                progress_value_int = max(0, min(100, int(percent)))
                st.progress(progress_value_int / 100.0)
                st.text(f"{progress_value_int}%")
            elif status_value == "completed":
                st.success("✅ Processing Complete!")
                st.progress(1.0)
            elif status_value == "completed_empty":
                 st.warning("⚠️ Processing finished, but no content chunks were generated.")
                 st.progress(1.0)
            else:
                st.error(f"❌ Processing Failed/Error: {message}")
                progress_value_int = max(0, min(100, int(percent)))
                if progress_value_int >= 0:
                    st.progress(progress_value_int / 100.0)
                    st.text(f"{progress_value_int}%")

            if st.session_state.debug_mode:
                st.json(data)


    st.sidebar.header("LLM Parameters")
    st.session_state.llm_temp = st.sidebar.slider(
        "Temperature", min_value=0.0, max_value=2.0, value=st.session_state.llm_temp, step=0.05,
        help="Controls randomness. Lower values make the output more deterministic."
    )
    st.session_state.llm_max_tokens = st.sidebar.number_input(
        "Max Tokens", min_value=50, max_value=4096, value=st.session_state.llm_max_tokens, step=64,
        help="Maximum number of tokens the model should generate in its response."
    )


    st.sidebar.header("Query Mode")

    graph_toggle_disabled = st.session_state.skip_graph_build or not (st.session_state.progress_data and st.session_state.progress_data.get("status") == "completed")
    if st.session_state.skip_graph_build and st.session_state.use_graph_rag:
        st.session_state.use_graph_rag = False
        st.sidebar.warning("Graph RAG disabled because graph build was skipped during processing.")

    st.session_state.use_graph_rag = st.sidebar.toggle(
        "Use Graph RAG",
        value=st.session_state.use_graph_rag,
        disabled=graph_toggle_disabled,
        help="Enable to use Graph RAG (Cypher) for answering questions. Disable for standard vector search RAG. (Requires graph build during processing)"
    )


    st.header("2. Chat with Document")

    processing_complete = (st.session_state.progress_data and
                          st.session_state.progress_data.get("status") == "completed")

    if not processing_complete:
        st.info("Please upload and process a document successfully before chatting.")
    else:

        chat_container = st.container(height=900)
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

                    if message.get("sources") and not st.session_state.use_graph_rag:
                        with st.expander("Sources"):
                             for i, source_data in enumerate(message["sources"]):
                                 content = source_data.get("content", "N/A")
                                 metadata = source_data.get("metadata", {})

                                 st.text_area(f"Source Chunk {i+1}", content, height=100, key=f"src_{message['timestamp']}_{i}")
                                 if st.session_state.debug_mode and metadata:
                                     st.json(metadata, key=f"meta_{message['timestamp']}_{i}")



        chat_input_disabled = not processing_complete
        user_question = st.chat_input("Ask a question about the document...", key="chat_input_box", disabled=chat_input_disabled)

        if user_question:
            st.session_state.user_question = user_question


            user_msg_timestamp = time.time()
            st.session_state.chat_history.append({
                "role": "user",
                "content": st.session_state.user_question,
                "timestamp": user_msg_timestamp
            })


            with st.spinner("Thinking..."):
                response_data = call_chat_api(
                    question=user_question,
                    use_graph=st.session_state.use_graph_rag,
                    temperature=st.session_state.llm_temp,
                    max_tokens=st.session_state.llm_max_tokens
                )


            print(f"DEBUG FRONTEND: API call completed with response: {response_data}")
            assistant_msg_timestamp = datetime.now().isoformat()
            print(f"DEBUG FRONTEND: Received response_data: {response_data}")

            if response_data:
                answer = response_data.get("answer", "Sorry, I couldn't find an answer.")
                sources = response_data.get("sources", [])
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "timestamp": assistant_msg_timestamp
                })
            else:
                 st.session_state.chat_history.append({
                     "role": "assistant",
                     "content": "Error communicating with the backend.",
                     "timestamp": assistant_msg_timestamp
                 })

            st.session_state.user_question = ""
            st.rerun()


    if needs_rerun_for_poll:
        time.sleep(0.5)
        st.rerun()


if __name__ == "__main__":
    main()