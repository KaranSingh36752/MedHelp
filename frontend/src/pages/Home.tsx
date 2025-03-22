import React, { useState, useEffect } from "react";
import { Box, Typography, Button, CircularProgress } from "@mui/material";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const Home: React.FC = () => {
  const [selectedPDF, setSelectedPDF] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [translatedChunks, setTranslatedChunks] = useState<string[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    const savedTranslation = localStorage.getItem("translatedChunks");
    if (savedTranslation) setTranslatedChunks(JSON.parse(savedTranslation));
  }, []);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) setSelectedPDF(file);
  };

  const handleFileUpload = async () => {
    if (!selectedPDF) {
      alert("Upload Pdf file here");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedPDF);

    setLoading(true);

    try {
      const { data } = await axios.post(
        "http://127.0.0.1:8000/translate-pdf/",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      setTranslatedChunks(data.translated_chunks);
      localStorage.setItem("translatedChunks", JSON.stringify(data.translated_chunks));
      alert("File processed successfully! Click OK to view Translated File.");
    } catch (error) {
      console.error("Error uploading file:", error);
      alert("Failed to process the file. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleClearTranslation = () => {
    setTranslatedChunks([]);
    localStorage.removeItem("translatedChunks");
  };

  return (
    <Box className="flex justify-center items-center h-[calc(100vh-64px)] bg-gray-900">
      {loading ? (
        <CircularProgress color="primary" size={80} />
      ) : (
        <Box className="bg-gray-800 rounded-2xl shadow-2xl w-3/4 h-3/4 min-w-[300px] p-10 flex flex-col justify-evenly items-center overflow-y-auto gap-10">
          <Box className="text-center">
            <Typography variant="h3" fontWeight="bold" className="mb-4 text-blue-400">
              WELCOME TO
            </Typography>
            <Typography variant="h4" fontWeight="bold" className="text-blue-300">
              LEGAL-DOC-TRANSLATE-QUERY-ASSISTANT PORTAL
            </Typography>
          </Box>

          <Typography className="text-gray-400 text-center max-w-2xl">
            "LegalDoc-Translate-Query-Assistant Portal enables users to quickly translate legal documents into English and get accurate, context-driven answers via an AI-powered chatbot."
          </Typography>

          <Box className="flex justify-center gap-8">
            <Button
              variant="contained"
              component="label"
              className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg text-lg"
            >
              UPLOAD A PDF
              <input type="file" hidden accept="application/pdf" onChange={handleFileChange} />
            </Button>
            <Button
              variant="contained"
              onClick={handleFileUpload}
              className="bg-gray-600 hover:bg-gray-700 text-white px-8 py-3 rounded-lg text-lg"
            >
              TRANSLATE
            </Button>
          </Box>

          {selectedPDF && (
            <Typography className="text-gray-400 text-center mt-4">
              Selected PDF: {selectedPDF.name}
            </Typography>
          )}

          {translatedChunks.length > 0 && (
            <Box className="bg-gray-700 rounded-xl p-6 w-full">
              <Box className="flex justify-between items-center">
                <Typography variant="h5" fontWeight="bold" className="text-blue-400">
                  Translated PDF
                </Typography>
                <Box className="flex gap-4">
                  <Button
                    variant="contained"
                    onClick={() => navigate("/assistant")}
                    className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg"
                  >
                    ASK ASSISTANT
                  </Button>
                  <Button
                    variant="contained"
                    onClick={handleClearTranslation}
                    className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg"
                  >
                    CLEAR
                  </Button>
                </Box>
              </Box>
              <Box className="text-gray-300 text-justify whitespace-pre-wrap mt-4">
                {translatedChunks.join("\n\n")}
              </Box>
            </Box>
          )}
        </Box>
      )}
    </Box>
  );
};

export default Home;
