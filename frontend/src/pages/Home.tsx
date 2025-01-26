import React from "react";
import { Box, Typography, Button } from "@mui/material";

const Home: React.FC = () => {
  return (
    <Box className="flex justify-center items-center h-[calc(100vh-64px)] bg-gray-900">
      <Box className="bg-gray-800 rounded-2xl shadow-2xl w-3/4 h-3/4 p-10 flex flex-col justify-evenly items-center">
        {/* Header Section */}
        <Box className="text-center">
          <Typography
            variant="h3"
            fontWeight="bold"
            className="mb-4 text-blue-400"
          >
            WELCOME TO
          </Typography>
          <Typography
            variant="h4"
            fontWeight="bold"
            className="text-blue-300"
          >
            LEGAL-DOC-TRANSLATE-QUERY-ASSISTANT PORTAL
          </Typography>
        </Box>

        {/* Description */}
        <Typography
          className="text-gray-400 text-center max-w-2xl"
        >
          "LegalDoc-Translate-Query-Assistant Portal enables users to quickly translate legal documents into English and get accurate, context-driven answers via an AI-powered chatbot." 
        </Typography>

        {/* Buttons */}
        <Box className="flex justify-center gap-8 mt-4">
          <Button
            variant="contained"
            className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg text-lg"
          >
            UPLOAD A FILE
          </Button>
          <Button
            variant="contained"
            className="bg-gray-600 hover:bg-gray-700 text-white px-8 py-3 rounded-lg text-lg"
          >
            TRANSLATE
          </Button>
        </Box>
      </Box>
    </Box>
  );
};

export default Home;
