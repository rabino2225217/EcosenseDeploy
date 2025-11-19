import * as React from "react";

//Different keys for client and admin
const CLIENT_THEME_KEY = "vite-ui-theme-client";
const ADMIN_THEME_KEY = "vite-ui-theme-admin";

type Theme = "dark" | "light";

interface UseThemeOptions {
  isAdmin?: boolean;
}

export function useTheme(options: UseThemeOptions = {}) {
  const { isAdmin = false } = options;

  // Choose the appropriate key based on context
  const THEME_KEY = isAdmin ? ADMIN_THEME_KEY : CLIENT_THEME_KEY;

  const [theme, setTheme] = React.useState<Theme>(() => {
    // Always default to light mode on page load/refresh
    return "light";
  });

  const root = window.document.documentElement;

  React.useEffect(() => {
    root.classList.remove("light", "dark");

    //Always apply light mode class
    root.classList.add("light");
  }, [root.classList]);

  const setThemeAndStore = React.useCallback(
    (newTheme: Theme) => {
      setTheme(newTheme);
      localStorage.setItem(THEME_KEY, newTheme);

      //Apply the selected theme immediately
      root.classList.remove("light", "dark");
      root.classList.add(newTheme);
    },
    [THEME_KEY, root.classList]
  );

  //Function to reset to light mode (call this on logout)
  const resetToLightMode = React.useCallback(() => {
    setTheme("light");
    localStorage.setItem(THEME_KEY, "light");

    root.classList.remove("light", "dark");
    root.classList.add("light");
  }, [THEME_KEY, root.classList]);

  return {
    theme,
    setTheme: setThemeAndStore,
    isDarkMode: theme === "dark",
    resetToLightMode,
  };
}
