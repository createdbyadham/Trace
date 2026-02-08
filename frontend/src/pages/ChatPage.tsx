"use client";

import { AnimatePresence, motion } from "framer-motion";
import { PlaceholdersAndVanishInput } from "../components/ui/placeholders-and-vanish-input";
import { useState, useEffect, useRef, useCallback } from "react";
import { streamChat, clearChat } from "../api/chat";
import { cn } from "@/lib/utils";
import { MeshGradient } from "../components/MeshGradient";
import { ReceiptScanOverlay } from "../components/ReceiptScanOverlay";
import { parseReceipt, type ScanResponse, type ParseResponse } from "@/api/ocr";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import Logo from "../../../Tracelogo.png"

interface Message {
  content: string;
  role: 'user' | 'assistant';
}

interface ScanState {
  imageUrl: string;
  scanResult: ScanResponse;
}

// Trace Logo Icon Component
const TraceIcon = ({ className = "" }: { className?: string }) => (
  <img src={Logo} alt="Trace Logo" className={className} />
);

// Placeholders for the input
const placeholders = [
  "How much did I spend on groceries last month?",
  "Show me all my coffee purchases",
  "Which store do I spend the most at?",
  "What was my most expensive purchase?",
  "Breakdown my spending by category",
];

export function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isExpanded, setIsExpanded] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [scanState, setScanState] = useState<ScanState | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Ref to hold the LLM parse promise that runs in parallel with the animation
  const parsePromiseRef = useRef<Promise<ParseResponse> | null>(null);
  const animationDoneRef = useRef(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);


  const handleChange = () => {
    // Handle input change if needed
  };

  const handleClearChat = async () => {
    try {
      await clearChat();
      setMessages([]);
      setIsExpanded(false);
    } catch (error) {
      console.error('Failed to clear chat:', error);
    }
  };

  // Called after the fast OCR scan completes — shows overlay AND fires LLM parse in parallel
  const handleReceiptScanned = useCallback((imageUrl: string, scanResult: ScanResponse) => {
    // 1. Show the scanning animation overlay
    setScanState({ imageUrl, scanResult });
    animationDoneRef.current = false;

    // 2. Fire LLM parse in the background (runs while animation plays)
    parsePromiseRef.current = parseReceipt(scanResult.raw_text);
  }, []);

  // Called when the scan overlay animation finishes
  const handleScanComplete = useCallback(async () => {
    if (!scanState) return;
    const { imageUrl } = scanState;

    // Clean up
    URL.revokeObjectURL(imageUrl);
    setScanState(null);
    animationDoneRef.current = true;

    // Add user message and expand chat
    setMessages(prev => [...prev, { content: "Uploaded a receipt for processing", role: 'user' as const }]);
    if (!isExpanded) setIsExpanded(true);

    // Wait for the LLM parse that was already running in parallel
    try {
      await parsePromiseRef.current;
      parsePromiseRef.current = null;

    } catch (error) {
      console.error('Failed to process receipt:', error);
    }
  }, [scanState, isExpanded]);

  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const message = formData.get("message") as string;
    const ocrData = formData.get("ocr_data") as string;

    if (!message?.trim() && !ocrData) return;

    // Add user message
    const userMessage = { content: message || "Uploaded a receipt for processing", role: 'user' as const };
    setMessages(prev => [...prev, userMessage]);

    if (!isExpanded) {
      setIsExpanded(true);
    }

    // Add a placeholder assistant message that will be streamed into
    setMessages(prev => [...prev, { content: '', role: 'assistant' as const }]);
    setIsStreaming(true);

    try {
      await streamChat(
        ocrData || message,
        // onToken — append each token to the last (assistant) message
        (token) => {
          setMessages(prev => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last && last.role === 'assistant') {
              updated[updated.length - 1] = { ...last, content: last.content + token };
            }
            return updated;
          });
        },
        // onDone
        () => {
          setIsStreaming(false);
        },
        // onError
        (error) => {
          console.error('Streaming error:', error);
          setIsStreaming(false);
        },
      );
    } catch (error) {
      console.error('Failed to send message:', error);
      setIsStreaming(false);
    }
  };

  return (
    <>
      <MeshGradient />
      <div className="fixed inset-0 flex flex-col w-full overflow-hidden z-10">
        {/* Header - fixed to top-left regardless of scroll */}
        <header className="sticky top-0 z-50 flex items-center justify-between px-6 py-4 flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-white rounded-md flex items-center justify-center">
              <TraceIcon className="w-6 h-6 text-[#0d1117]" />
            </div>
            <span className="text-white font-medium text-lg">Trace</span>
          </div>

          {/* Clear Chat Button - only visible when chat is expanded */}
          <AnimatePresence>
            {isExpanded && (
              <motion.button
                onClick={handleClearChat}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#2d333d]/80 hover:bg-[#3d444d] text-gray-300 hover:text-white transition-colors duration-200"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.2 }}
                title="Clear chat history"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="w-4 h-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                  />
                </svg>
                <span className="text-sm font-medium">Clear</span>
              </motion.button>
            )}
          </AnimatePresence>
        </header>

        {/* Main Content */}
        <div className={cn(
          "flex flex-col flex-1 min-h-0 px-4",
          !isExpanded && "justify-center items-center"
        )}>
          {/* Welcome Section - Only shown when not expanded */}
          {!isExpanded && (
            <motion.div
              className="flex flex-col items-center text-center mb-8"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              {/* Logo Icon */}
              <div className="w-16 h-16 bg-white rounded-2xl flex items-center justify-center mb-8 shadow-lg">
                <TraceIcon className="w-12 h-12 text-[#1a2332]" />
              </div>

              {/* Greeting */}
              <h2 className="text-[#a1aace] text-2xl font-medium mb-2">
                Good Morning, Adham!
              </h2>
              <h1 className="text-white text-3xl sm:text-3xl font-semibold mb-4">
                How may I assist you today?
              </h1>

              {/* Subtitle */}
              <p className="text-[#a1aace] text-xs max-w-md leading-relaxed">
                Ready to assist you with anything you need regarding your spendings.
                <br />
                From answering questions, to providing recommendations.
              </p>
            </motion.div>
          )}

          {/* Chat Messages - Only shown when expanded */}
          <AnimatePresence>
            {isExpanded && (
              <motion.div
                className="flex-1 overflow-y-auto min-h-0 mb-4 w-full scroll-smooth"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
              >
                <div className="max-w-[46.5%] mx-auto space-y-4">
                  {messages.filter(m => m.content.length > 0).map((message, index) => (
                    <motion.div
                      key={index}
                      className={cn(
                        "flex",
                        message.role === 'user' ? "justify-end" : "justify-start"
                      )}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      <div className={cn(
                        "px-4 py-2 rounded-2xl max-w-[85%] sm:max-w-[75%]",
                        message.role === 'user'
                          ? "bg-[#2d333d] text-white"
                          : "bg-[#353f4c] backdrop-blur-sm text-gray-100"
                      )}>
                        {message.role === 'assistant' ? (
                          <div className="prose prose-invert max-w-none text-left">
                            <ReactMarkdown
                              remarkPlugins={[remarkGfm]}
                              rehypePlugins={[rehypeRaw]}
                              components={{
                                p: ({ ...props }) => <p className="m-0" {...props} />,
                                ul: ({ ...props }) => <ul className="m-0 pl-4" {...props} />,
                                ol: ({ ...props }) => <ol className="m-0 pl-4" {...props} />,
                                li: ({ ...props }) => <li className="my-1" {...props} />,
                                strong: ({ ...props }) => <strong className="font-bold text-blue-300" {...props} />,
                                h1: ({ ...props }) => <h1 className="text-xl font-bold my-2" {...props} />,
                                h2: ({ ...props }) => <h2 className="text-lg font-bold my-2" {...props} />,
                                h3: ({ ...props }) => <h3 className="text-base font-bold my-1" {...props} />,
                                code: ({ ...props }) => <code className="bg-black/30 rounded px-1" {...props} />,
                                table: ({ ...props }) => <table className="border-collapse my-2" {...props} />,
                                th: ({ ...props }) => <th className="border border-gray-600 px-2 py-1" {...props} />,
                                td: ({ ...props }) => <td className="border border-gray-600 px-2 py-1" {...props} />
                              }}
                            >
                              {message.content}
                            </ReactMarkdown>
                          </div>
                        ) : (
                          message.content
                        )}
                      </div>
                    </motion.div>
                  ))}
                  {isStreaming && messages[messages.length - 1]?.content === '' && (
                    <motion.div
                      className="flex justify-start"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 10 }}
                    >
                      <div className="px-4 py-2 rounded-3xl bg-zinc-800/70 backdrop-blur-sm">
                        <div className="flex items-center space-x-1">
                          <motion.div
                            className="w-1.5 h-1.5 bg-gray-300 rounded-full"
                            animate={{ scale: [1, 1.2, 1] }}
                            transition={{ duration: 1, repeat: Infinity, delay: 0 }}
                          />
                          <motion.div
                            className="w-1.5 h-1.5 bg-gray-300 rounded-full"
                            animate={{ scale: [1, 1.2, 1] }}
                            transition={{ duration: 1, repeat: Infinity, delay: 0.2 }}
                          />
                          <motion.div
                            className="w-1.5 h-1.5 bg-gray-300 rounded-full"
                            animate={{ scale: [1, 1.2, 1] }}
                            transition={{ duration: 1, repeat: Infinity, delay: 0.4 }}
                          />
                        </div>
                      </div>
                    </motion.div>
                  )}
                  <div ref={messagesEndRef} />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Input Section */}
          <div className={cn(
            "w-full max-w-2xl mx-auto flex-shrink-0 flex flex-col items-center gap-4",
            !isExpanded && "w-full"
          )}>

            {/* Input Field */}
            <PlaceholdersAndVanishInput
              placeholders={placeholders}
              onChange={handleChange}
              onSubmit={onSubmit}
              onReceiptScanned={handleReceiptScanned}
            />
          </div>
        </div>

        {/* Footer */}
        <footer className="px-6 py-4 text-center flex-shrink-0">
          <p className="text-gray-500 text-xs">
            Your data remains local and isn't uploaded to any server.
          </p>
        </footer>

        {/* Receipt scanning overlay */}
        {scanState && (
          <ReceiptScanOverlay
            imageUrl={scanState.imageUrl}
            textRegions={scanState.scanResult.ocr_regions.text_regions}
            imageWidth={scanState.scanResult.ocr_regions.image_width}
            imageHeight={scanState.scanResult.ocr_regions.image_height}
            onComplete={handleScanComplete}
          />
        )}
      </div>
    </>
  );
}
