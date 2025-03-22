import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";

import AppBar from "@mui/material/AppBar";
import Box from "@mui/material/Box";
import CssBaseline from "@mui/material/CssBaseline";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";

import Home from "../pages/Home";
import QueryAssistant from "../pages/QueryAssistant";

const navItems = [
  { label: "Home", path: "/" },
  { label: "Assistant", path: "/assistant" },
];

export default function Navbar() {
  return (
    <Router>
      <Box className="flex flex-col h-screen">
        <CssBaseline />
        <AppBar component="nav" className="bg-gray-900">
          <Toolbar>
            <Typography
              variant="h6"
              component="div"
              className="flex-grow text-blue-400"
            >
              MedHelp Gpt
            </Typography>
            <Box className="hidden sm:flex">
              {navItems.map((item) => (
                <Button
                  key={item.label}
                  component={Link}
                  to={item.path}
                  className="text-white hover:bg-blue-600"
                >
                  {item.label}
                </Button>
              ))}
            </Box>
          </Toolbar>
        </AppBar>
        <Box
          component="main"
          className="p-6 w-full bg-gray-900 text-white flex-grow overflow-auto"
        >
          <Toolbar />
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/assistant" element={<QueryAssistant />} />
          </Routes>
        </Box>
      </Box>
    </Router>
  );
}
