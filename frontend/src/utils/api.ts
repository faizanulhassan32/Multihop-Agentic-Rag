// API utilities for LangGraph communication and database management

// Simplified LangGraph client for direct HTTP communication
export const getLangGraphClient = () => {
  const baseUrl = 'http://localhost:2024';

  return {
    assistants: {
      search: async (): Promise<any[]> => {
        try {
          const response = await fetch(`${baseUrl}/assistants`);
          if (!response.ok) throw new Error('Failed to fetch assistants');
          return await response.json();
        } catch (error) {
          console.error('Error fetching assistants:', error);
          return [];
        }
      }
    },
    
    threads: {
      create: async (): Promise<{ thread_id: string }> => {
        try {
          // Send an empty JSON object as the body to avoid zero-length error
          const response = await fetch(`${baseUrl}/threads`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
          });
          if (!response.ok) throw new Error('Failed to create thread');
          return await response.json();
        } catch (error) {
          console.error('Error creating thread:', error);
          throw error;
        }
      }
    },

    runs: {
      stream: async function* (
        threadId: string, 
        assistantId: string, 
        options: { input: any; config: any; stream_mode: string }
      ): AsyncIterableIterator<any> {
        // This is kept for compatibility but we use polling instead
        throw new Error('Streaming not implemented - use polling approach instead');
      }
    }
  };
};


// LangGraph server health check using /ok endpoint and env variable
export const checkLangGraphStatus = async (): Promise<boolean> => {
  try {
    const apiUrl = import.meta.env.VITE_LANGGRAPH_URL || 'http://localhost:2024';
    const response = await fetch(`${apiUrl}/ok`, {
      method: 'GET',
      mode: 'cors'
    });
    if (response.ok) {
      const data = await response.json();
      return data.ok === true;
    }
    return false;
  } catch (error) {
    console.error('Error checking LangGraph status:', error);
    return false;
  }
};

// Main chat function that replicates the Streamlit logic
export const sendChatMessage = async (
  message: string,
  model: string,
  onProgress?: (progress: string) => void
): Promise<string> => {
  try {
    const client = getLangGraphClient();
    const assistantId = import.meta.env.VITE_ASSISTANT_ID;
    if (!assistantId) {
      console.error('[sendChatMessage] No assistant ID found in env.');
      throw new Error('Assistant ID is not set. Please set VITE_ASSISTANT_ID in your .env file.');
    }
    console.log('[sendChatMessage] Using assistant ID:', assistantId);
    console.log('[sendChatMessage] Creating thread...');
    const thread = await client.threads.create();
    console.log('[sendChatMessage] Thread created:', thread);

    // Prepare input and config
    const inputState = { original_question: message };
    const config = {
      configurable: {
        model: model,
        temperature: 0.3,
        max_tokens: 500,
      }
    };
    console.log('[sendChatMessage] Input state:', inputState);
    console.log('[sendChatMessage] Config:', config);

    // Try synchronous request first (simpler approach)
    console.log('[sendChatMessage] Trying synchronous request...');
    try {
      const syncResponse = await fetch('http://localhost:2024/threads/' + thread.thread_id + '/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          assistant_id: assistantId,
          input: inputState,
          config: config
          // Note: No stream_mode for synchronous request
        })
      });

      console.log('[sendChatMessage] Sync request status:', syncResponse.status);
      console.log('[sendChatMessage] Sync request headers:', Object.fromEntries(syncResponse.headers.entries()));

      if (syncResponse.ok) {
        const result = await syncResponse.json();
        console.log('[sendChatMessage] Synchronous response:', result);

        // Check if we got a complete result
        if (result && result.final_answer) {
          console.log('[sendChatMessage] Got final answer from sync request:', result.final_answer);
          return result.final_answer;
        }

        // If the result has a run_id, poll for completion
        if (result && result.run_id) {
          console.log('[sendChatMessage] Got run_id, polling for completion:', result.run_id);

          // Poll for completion
          const pollInterval = 2000; // 2 seconds
          const maxPolls = 30; // Maximum 60 seconds

          for (let pollCount = 0; pollCount < maxPolls; pollCount++) {
            try {
              console.log(`[sendChatMessage] Polling attempt ${pollCount + 1}/${maxPolls}`);

              const threadStateResponse = await fetch(`http://localhost:2024/threads/${thread.thread_id}/state`);
              if (threadStateResponse.ok) {
                const threadState = await threadStateResponse.json();
                console.log('[sendChatMessage] Polled thread state:', threadState);

                if (threadState.values && threadState.values.final_answer) {
                  const finalAnswer = threadState.values.final_answer;
                  console.log('[sendChatMessage] Found final answer via polling:', finalAnswer);

                  if (onProgress) {
                    onProgress(`✅ Final Answer: ${finalAnswer}`);
                  }

                  return finalAnswer;
                }

                // Show progress based on current state
                if (onProgress && threadState.values) {
                  const state = threadState.values;
                  if (state.decomposed_questions && state.decomposed_questions.length > 0) {
                    const currentStep = (state.current_question_index || 0) + 1;
                    const totalSteps = state.decomposed_questions.length;
                    onProgress(`🧠 Decomposed into ${totalSteps} questions. Processing step ${currentStep}/${totalSteps}...`);
                  }

                  if (state.intermediate_answers && state.intermediate_answers.length > 0) {
                    const answers = state.intermediate_answers;
                    const latestAnswer = answers[answers.length - 1];
                    onProgress(`💭 Gathering information: ${latestAnswer.substring(0, 100)}...`);
                  }
                }
              }

              // Wait before next poll
              await new Promise(resolve => setTimeout(resolve, pollInterval));
            } catch (error) {
              console.warn(`[sendChatMessage] Poll attempt ${pollCount + 1} failed:`, error);
            }
          }
        }
      } else {
        const errorText = await syncResponse.text();
        console.error('[sendChatMessage] Sync request failed:', syncResponse.status, errorText);
      }
    } catch (syncError) {
      console.warn('[sendChatMessage] Synchronous request failed:', syncError);
    }

    // If we get here, both sync and polling failed
    console.warn('[sendChatMessage] All methods failed to get response');
    return "I apologize, but I couldn't generate a complete response. Please try again.";

  } catch (error) {
    console.error('[sendChatMessage] Error:', error);
    throw new Error(`An error occurred: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
};