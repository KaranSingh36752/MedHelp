import React, { useState } from "react";
import { Box, Typography, Button } from "@mui/material";
import axios from "axios";

const Home: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // Handle file selection
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
    }
  };

  // Handle file upload to the backend
  const handleFileUpload = async () => {
    if (!selectedFile) {
      alert("Please select a PDF file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post("https://f333-34-90-206-123.ngrok-free.app/translate-pdf/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      alert("File processed successfully! Check console for details.");
      console.log(response.data);
    } catch (error) {
      console.error("Error uploading file:", error);
      alert("Failed to process the file. Please try again.");
    }
  };

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
        <Typography className="text-gray-400 text-center max-w-2xl">
          "LegalDoc-Translate-Query-Assistant Portal enables users to quickly translate legal documents into English and get accurate, context-driven answers via an AI-powered chatbot."
        </Typography>

        {/* Buttons */}
        <Box className="flex justify-center gap-8 mt-4">
          {/* Upload a PDF Button */}
          <Button
            variant="contained"
            component="label"
            className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg text-lg"
          >
            UPLOAD A PDF
            <input
              type="file"
              hidden
              accept="application/pdf"
              onChange={handleFileChange}
            />
          </Button>

          {/* Translate Button */}
          <Button
            variant="contained"
            onClick={handleFileUpload}
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
