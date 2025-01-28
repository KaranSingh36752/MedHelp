import React, { useState, useRef, useEffect } from "react";
import { Box, Typography, TextField, Button } from "@mui/material";
import axios from "axios";

const ChatScreen: React.FC = () => {
  const [messages, setMessages] = useState<{ sender: string; text: string }[]>([]);
  const [currentMessage, setCurrentMessage] = useState("");
  const chatWindowRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatWindowRef.current?.scrollTo({ top: chatWindowRef.current.scrollHeight });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!currentMessage.trim()) return;

    setMessages((prev) => [...prev, { sender: "User", text: currentMessage }]);
    setCurrentMessage("");

    try {
      const { data } = await axios.post("https://4b00-34-169-3-56.ngrok-free.app/query/", { user_query: currentMessage });
      const formattedResponse = `Answer: ${data.answer}\n\nReasoning: ${data.reasoning}\n\nConfidence Score ${data.confidence_score}`;
      setMessages((prev) => [...prev, { sender: "AI", text: formattedResponse }]);
    } catch {
      setMessages((prev) => [...prev, { sender: "AI", text: "Error processing query." }]);
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === "Enter") {
      event.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <Box className="flex justify-center items-center h-[calc(100vh-64px)] bg-gray-900">
      <Box className="bg-gray-800 rounded-2xl shadow-2xl w-3/4 h-3/4 min-w-[300px] p-6 flex flex-col">
        {/* Header */}
        <Box className="flex justify-between items-center mb-4">
          <Typography variant="h4" fontWeight="bold" className="text-blue-300">
            LEGAL-QUERY-ASSISTANT
          </Typography>
          <Button
            variant="contained"
            onClick={() => (window.location.href = "/")}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg"
          >
            HOME
          </Button>
        </Box>

        {/* Chat Window */}
        <Box ref={chatWindowRef} className="flex-1 bg-gray-700 rounded-xl p-4 overflow-y-auto">
          {messages.map((message, index) => (
            <Box key={index} className={`mb-4 ${message.sender === "User" ? "text-right" : "text-left"}`}>
              <Typography
                className={`inline-block px-4 py-2 rounded-lg text-white ${
                  message.sender === "User" ? "bg-blue-600" : "bg-gray-600"
                }`}
              >
                {message.text.split("\n").map((line, idx) => (
                  <span key={idx}>{line}<br /></span>
                ))}
              </Typography>
            </Box>
          ))}
        </Box>

        {/* Input Section */}
        <Box className="mt-4 flex items-center gap-4">
          <TextField
            variant="outlined"
            placeholder="Type Something..."
            value={currentMessage}
            onChange={(e) => setCurrentMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            className="flex-1 bg-gray-900 rounded-lg text-gray-300"
            InputProps={{ style: { color: "white" } }}
          />
          <Button
            variant="contained"
            onClick={handleSendMessage}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg"
          >
            Send
          </Button>
        </Box>
      </Box>
    </Box>
  );
};

export default ChatScreen;
