import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Card } from '@/components/ui/card';
import { SampleQuestions } from './SampleQuestions';
import { Instructions } from './Instructions';
import { ModelSelector } from './ModelSelector';
import { useToast } from '@/hooks/use-toast';
import { sendChatMessage, checkLangGraphStatus } from '@/utils/api';
import TraceTimeline from './TraceTimeline';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  trace?: any[];
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  lastUpdated: Date;
}

const ChatInterface = () => {
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [selectedModel, setSelectedModel] = useState('llama3:latest');
  const [langGraphStatus, setLangGraphStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [progressMessage, setProgressMessage] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [currentSession?.messages]);

  useEffect(() => {
    checkStatuses();
  }, []);

  const checkStatuses = async () => {

  // Check LangGraph status
    try {
      const isConnected = await checkLangGraphStatus();
      setLangGraphStatus(isConnected ? 'connected' : 'disconnected');
    } catch (error) {
      setLangGraphStatus('disconnected');
    }
  };

  const createNewChat = () => {
    const newSession: ChatSession = {
      id: Date.now().toString(),
      title: 'New Chat',
      messages: [],
      createdAt: new Date(),
      lastUpdated: new Date(),
    };
    
    setChatSessions(prev => [newSession, ...prev]);
    setCurrentSession(newSession);
  };

  const selectChatSession = (sessionId: string) => {
    const session = chatSessions.find(s => s.id === sessionId);
    if (session) {
      setCurrentSession(session);
    }
  };

  const deleteChatSession = (sessionId: string) => {
    setChatSessions(prev => prev.filter(s => s.id !== sessionId));
    if (currentSession?.id === sessionId) {
      const remainingSessions = chatSessions.filter(s => s.id !== sessionId);
      setCurrentSession(remainingSessions.length > 0 ? remainingSessions[0] : null);
    }
    toast({
      title: "Chat deleted",
      description: "The chat session has been removed.",
    });
  };

  // ...existing code...

  const handleSampleQuestion = (question: string) => {
    setMessage(question);
  };

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!message.trim()) return;
    

    if (langGraphStatus !== 'connected') {
      toast({
        title: "LangGraph not connected",
        description: "Please ensure the LangGraph server is running.",
        variant: "destructive",
      });
      return;
    }

    if (!currentSession) {
      createNewChat();
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: message,
      timestamp: new Date(),
    };

    const updatedSession = {
      ...currentSession!,
      messages: [...(currentSession?.messages || []), userMessage],
      title: currentSession?.messages.length === 0 ? message.slice(0, 50) + '...' : currentSession!.title,
      lastUpdated: new Date(),
    };

    setCurrentSession(updatedSession);
    setChatSessions(prev => prev.map(s => s.id === updatedSession.id ? updatedSession : s));
    setMessage('');
    setIsLoading(true);
    setProgressMessage('🕵️‍♂️ The agent is on the case... This might take a moment.');

    try {
      const response = await sendChatMessage(
        userMessage.content, 
        selectedModel,
        (progress: string) => {
          setProgressMessage(progress);
        }
      );
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.final_answer,
        trace: response.trace,
        timestamp: new Date(),
      };

      const finalSession = {
        ...updatedSession,
        messages: [...updatedSession.messages, assistantMessage],
        lastUpdated: new Date(),
      };

      setCurrentSession(finalSession);
      setChatSessions(prev => prev.map(s => s.id === finalSession.id ? finalSession : s));
      
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to get response from the agent.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
      setProgressMessage('');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-hero flex flex-col">
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar - New Chat */}
        <div className="w-64 bg-card border-r border-border p-4 flex flex-col">
          <Button 
            onClick={createNewChat}
            className="w-full bg-gradient-primary hover:shadow-glow transition-all duration-300 mb-4"
          >
            New Chat
          </Button>
          
          <div className="flex-1">
            <h3 className="text-sm font-medium text-muted-foreground mb-3">Recent Chats</h3>
            <div className="space-y-2">
              {chatSessions.map((session) => (
                <div
                  key={session.id}
                  className={`p-3 rounded-lg cursor-pointer transition-colors ${
                    currentSession?.id === session.id
                      ? 'bg-primary/10 border border-primary/20'
                      : 'bg-card/50 hover:bg-card/80'
                  }`}
                  onClick={() => selectChatSession(session.id)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{session.title}</p>
                      <p className="text-xs text-muted-foreground">
                        {session.lastUpdated.toLocaleDateString()}
                      </p>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteChatSession(session.id);
                      }}
                      className="opacity-0 group-hover:opacity-100 transition-opacity ml-2"
                    >
                      <X className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <div className="p-6 border-b border-border">
            <h1 className="text-3xl font-bold bg-gradient-primary bg-clip-text text-transparent">
              Multi-Hop Agentic RAG
            </h1>
          </div>

          {/* Chat Messages */}
          <ScrollArea className="flex-1 p-6">
            {currentSession && currentSession.messages.length > 0 ? (
              <div className="space-y-4 max-w-4xl mx-auto">
                {currentSession.messages.map((msg) => (
                  <Card 
                    key={msg.id} 
                    className={`p-4 ${
                      msg.role === 'user' 
                        ? 'ml-auto max-w-2xl bg-primary/10 border-primary/20' 
                        : 'mr-auto max-w-3xl bg-gradient-card border-border'
                    } shadow-card animate-slide-up`}
                  >
                    <div className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                        msg.role === 'user' 
                          ? 'bg-primary text-primary-foreground' 
                          : 'bg-secondary text-secondary-foreground'
                      }`}>
                        {msg.role === 'user' ? 'U' : 'AI'}
                      </div>
                      <div className="flex-1 space-y-2">
                        <div className="text-sm text-muted-foreground">
                          {msg.timestamp.toLocaleTimeString()}
                        </div>
                        <div className="prose prose-invert max-w-none">
                          {msg.content}
                        </div>
                        {msg.role === 'assistant' && Array.isArray(msg.trace) && msg.trace.length > 0 && (
                          <TraceTimeline trace={msg.trace} />
                        )}
                      </div>
                    </div>
                  </Card>
                ))}
                {isLoading && (
                  <Card className="mr-auto max-w-3xl bg-gradient-card border-border shadow-card">
                    <div className="p-4 flex items-center gap-3">
                      <div className="w-8 h-8 bg-secondary rounded-full flex items-center justify-center">
                        <Loader2 className="w-4 h-4 animate-spin text-secondary-foreground" />
                      </div>
                      <div className="flex-1">
                        <div className="text-muted-foreground mb-1">Investigating...</div>
                        {progressMessage && (
                          <div className="text-sm text-primary animate-pulse">
                            {progressMessage}
                          </div>
                        )}
                      </div>
                    </div>
                  </Card>
                )}
                <div ref={messagesEndRef} />
              </div>
            ) : (
              <div className="flex-1 flex items-center justify-center">
                <div className="max-w-2xl text-center space-y-6">
                  <h2 className="text-2xl font-semibold text-foreground">
                    Welcome to Multi-Hop Agentic RAG
                  </h2>
                  <p className="text-muted-foreground">
                    Ask complex questions and let our AI agent investigate across multiple sources to find the answers.
                  </p>
                  <SampleQuestions onQuestionSelect={handleSampleQuestion} />
                </div>
              </div>
            )}
          </ScrollArea>

          {/* Chat Input */}
          <div className="p-6 border-t border-border bg-card/30 backdrop-blur-sm">
            <form onSubmit={sendMessage} className="max-w-4xl mx-auto">
              <div className="flex gap-2">
                <Input
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="Ask your multi-hop question..."
                  className="flex-1 bg-input border-border focus:border-primary focus:ring-primary/20"
                  disabled={isLoading || langGraphStatus !== 'connected'}
                />
                <Button 
                  type="submit" 
                  disabled={isLoading || !message.trim() || langGraphStatus !== 'connected'}
                  className="bg-gradient-primary hover:shadow-glow transition-all duration-300"
                >
                  {isLoading ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Send className="w-4 h-4" />
                  )}
                </Button>
              </div>
            </form>
          </div>
        </div>

        {/* Right Sidebar - Configuration & LangGraph Status */}
        <div className="w-80 bg-card border-l border-border p-4 flex flex-col space-y-6">
          {/* LangGraph Status */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-foreground">System Status</h3>
            <div className="space-y-2">
              <div className="flex items-center justify-between p-2 rounded-lg bg-card/50">
                <span className="text-xs text-muted-foreground">LangGraph</span>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${
                    langGraphStatus === 'connected' ? 'bg-success' : 'bg-destructive'
                  }`} />
                  <span className="text-xs">{
                    langGraphStatus === 'connected' ? 'Connected' : 'Disconnected'
                  }</span>
                </div>
              </div>
            </div>
            <Button 
              onClick={checkStatuses}
              variant="outline"
              size="sm"
              className="w-full text-xs"
            >
              Refresh Status
            </Button>
          </div>

          {/* Model Selection */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-foreground">Configuration</h3>
            <ModelSelector 
              value={selectedModel}
              onChange={setSelectedModel}
            />
          </div>

          {/* Instructions */}
          <div className="flex-1">
            <Instructions />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;