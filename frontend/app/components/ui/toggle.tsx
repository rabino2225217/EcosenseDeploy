// app/components/ui/toggle.tsx
import * as React from "react";
import { Sun, Moon } from "lucide-react";
import { useTheme } from "../../lib/use-darkmode";

export function ThemeToggle() {
  const { isDarkMode, setTheme } = useTheme();

  const toggleTheme = () => {
    setTheme(isDarkMode ? "light" : "dark");
  };

  return (
    <button
      type="button"
      onClick={toggleTheme}
      aria-label={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}
      className={`
        inline-flex items-center justify-center
        w-8 h-8 rounded-full transition-all duration-300
        ${
          isDarkMode
            ? "bg-gray-950 text-gray-100 hover:bg-gray-700"
            : "bg-gray-200 text-gray-700 hover:bg-gray-300"
        }
        hover:scale-110 focus:outline-none focus:ring-1 focus:ring-primary/50
      `}
    >
      {isDarkMode ? (
        <Moon className="w-4 h-4 transition-transform duration-300" />
      ) : (
        <Sun className="w-4 h-4 transition-transform duration-300" />
      )}
    </button>
  );
}
