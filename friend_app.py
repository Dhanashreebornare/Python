# 9. Process Active Client Message Inputs
if user_input := st.chat_input("Say something to Gaurav..."):
    with st.chat_message("user", avatar="✨"):
        st.markdown(user_input)
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.api_history.append(
        types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
    )

    # Generate response turn using active connection
    with st.chat_message("assistant", avatar="🧑‍💻"):
        message_placeholder = st.empty()
        token_placeholder = st.empty()
        
        full_response = ""
        token_string = ""
        api_success = False
        
        with st.spinner("Gaurav is typing... 💬"):
            try:
                # --- QUOTA MINIMIZER: Rolling Context Window ---
                MAX_HISTORY_TURNS = 6
                if len(st.session_state.api_history) > MAX_HISTORY_TURNS:
                    payload = st.session_state.api_history[-MAX_HISTORY_TURNS:]
                else:
                    payload = st.session_state.api_history

                # Fire structured API request
                response = generate_content_with_retry(payload)
                full_response = response.text
                
                # Extract token metrics safely from response metadata
                input_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
                output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
                token_string = f"⚡ Usage Check: {input_tokens} in | {output_tokens} out tokens"
                api_success = True
                
            except APIError as api_err:
                if api_err.code == 429:
                    st.error("🚨 **Gaurav is out of breath, bro!** The free limits ran out. Give him 15-20 seconds to catch his breath before typing again!")
                elif api_err.code == 503:
                    st.error("Gaurav's line is locked up due to high traffic! 😅 Try hitting send again in a few seconds.")
                else:
                    st.error(f"Gaurav hit a network snag: {api_err.message} (Status: {api_err.code})")
            except Exception as e:
                st.error(f"Gaurav went offline for a second. Try again! Details: {e}")

        # --- Smooth Fluid Typing Simulation Engine (Executed Safely Outside Try Block) ---
        if api_success and full_response:
            words = full_response.split(" ")
            for i in range(1, len(words) + 1):
                message_placeholder.markdown(" ".join(words[:i]) + " ▌")
                time.sleep(0.035) 
                
            # Final clean layout pass
            message_placeholder.markdown(full_response)
            token_placeholder.markdown(f"<span class='token-footer'>{token_string}</span>", unsafe_allow_html=True)
            
            # Save assistant output to visual UI log
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "token_info": token_string
            })
            
            # Save assistant output structural payload to API history log
            st.session_state.api_history.append(
                types.Content(role="model", parts=[types.Part.from_text(text=full_response)])
            )
