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
  onProgress?: (progress: string) => void,
  onToken?: (token: string) => void
): Promise<{ final_answer: string; trace: any[] }> => {
  try {
    const assistantId = import.meta.env.VITE_ASSISTANT_ID;
    if (!assistantId) {
      throw new Error('Assistant ID is not set. Please set VITE_ASSISTANT_ID in your .env file.');
    }

    // Create thread
    const threadResponse = await fetch('http://localhost:2024/threads', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });

    if (!threadResponse.ok) {
      throw new Error('Failed to create thread');
    }

    const thread = await threadResponse.json();

    // Prepare input and config (same as Streamlit)
    const inputState = { original_question: message };
    const config = {
      configurable: {
        model: model,
        temperature: 0.3,
        max_tokens: 500,
      }
    };

    // Use streaming approach like Streamlit does
    const streamResponse = await fetch(`http://localhost:2024/threads/${thread.thread_id}/runs/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        assistant_id: assistantId,
        input: inputState,
        config: config,
        stream_mode: "values"  // This is the key - matches Streamlit!
      })
    });

    if (!streamResponse.ok) {
      throw new Error(`Stream request failed: ${streamResponse.status}`);
    }

    // Process the streaming response like Streamlit does
    const reader = streamResponse.body?.getReader();
    if (!reader) {
      throw new Error('No response body reader available');
    }

    const decoder = new TextDecoder();
    let buffer = '';
    let fullResponse = '';
    let lastState: any = {};
    let lastTrace: any[] = [];

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.trim() === '') continue;

          try {
            // Parse SSE format (Server-Sent Events)
            if (line.startsWith('data: ')) {
              const data = JSON.parse(line.slice(6));

              // Handle token streaming events incrementally
              if (data.event === 'token' && data.data?.token) {
                const token = data.data.token;
                fullResponse += token;
                if (onToken) {
                  onToken(token);
                }
              }

              // Handle status progress events
              if (data.event === 'progress' && data.data?.message) {
                if (onProgress) {
                  onProgress(data.data.message);
                }
              }

              // Handle state updates and final answer completion
              if (data.event === 'values' && data.data) {
                const chunkData = data.data;

                if (typeof chunkData === 'object') {
                  lastState = { ...lastState, ...chunkData };
                  if (Array.isArray(chunkData.trace)) {
                    lastTrace = chunkData.trace;
                  }

                  if (chunkData.final_answer) {
                    fullResponse = chunkData.final_answer;
                    return { final_answer: fullResponse, trace: lastTrace };
                  }
                }
              }
            }
          } catch (parseError) {
            console.warn('Failed to parse streaming chunk:', parseError);
            continue;
          }
        }
      }
    } finally {
      reader.releaseLock();
    }

    // If we get here without finding a final answer, return the last known state
    if (fullResponse) {
      return { final_answer: fullResponse, trace: lastTrace };
    }

    // Fallback: check final thread state
    try {
      const finalStateResponse = await fetch(`http://localhost:2024/threads/${thread.thread_id}/state`);
      if (finalStateResponse.ok) {
        const finalState = await finalStateResponse.json();
        if (finalState.values?.final_answer) {
          return { final_answer: finalState.values.final_answer, trace: finalState.values?.trace || [] };
        }
      }
    } catch (error) {
      console.warn('Failed to get final state:', error);
    }

    return { final_answer: "I apologize, but I couldn't generate a complete response. Please try again.", trace: lastTrace };

  } catch (error) {
    console.error('[sendChatMessage] Error:', error);
    throw new Error(`An error occurred: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
};