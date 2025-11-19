import * as React from "react";

type AnalysisStatusContextValue = {
  analyzing: boolean;
};

const AnalysisStatusContext = React.createContext<AnalysisStatusContextValue>({
  analyzing: false,
});

export function AnalysisStatusProvider({ children }: { children: React.ReactNode }) {
  const [analyzing, setAnalyzing] = React.useState(false);

  React.useEffect(() => {
    const handler = (e: Event) => {
      const ce = e as CustomEvent<{ loading?: boolean }>;
      if (ce.detail && typeof ce.detail.loading === "boolean") {
        setAnalyzing(ce.detail.loading);
      }
    };

    window.addEventListener("analysis:status", handler as EventListener);
    return () =>
      window.removeEventListener("analysis:status", handler as EventListener);
  }, []);

  return (
    <AnalysisStatusContext.Provider value={{ analyzing }}>
      {children}
    </AnalysisStatusContext.Provider>
  );
}

export function useAnalysisStatus() {
  return React.useContext(AnalysisStatusContext);
}
