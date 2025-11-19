// routes/client/analysis-store.tsx
import * as React from "react";

type DroneImage = {
  file: File;
  url: string;
  uploadedAt: Date;
  selected: boolean;
  resultUrl?: string;
  detections?: { label: string; coordinates: any; gps_coordinates?: any }[];
};

type DetectionSettings = {
  confidence: number[];
  iou: number[];
};

// Separate stores
type AnalysisImagesStore = {
  [projectId: string]: DroneImage[];
};

type AnalysisSettingsStore = {
  [projectId: string]: DetectionSettings;
};

type CtxValue = {
  images: DroneImage[];
  setImages: React.Dispatch<React.SetStateAction<DroneImage[]>>;
  confidence: number[];
  setConfidence: (confidence: number[]) => void;
  iou: number[];
  setIou: (iou: number[]) => void;
};

const AnalysisStoreContext = React.createContext<{
  imagesStore: AnalysisImagesStore;
  setImagesStore: React.Dispatch<React.SetStateAction<AnalysisImagesStore>>;
  settingsStore: AnalysisSettingsStore;
  setSettingsStore: React.Dispatch<React.SetStateAction<AnalysisSettingsStore>>;
} | null>(null);

export function AnalysisStoreProvider({ children }: { children: React.ReactNode }) {
  // Images stored in memory only (not persisted)
  const [imagesStore, setImagesStore] = React.useState<AnalysisImagesStore>({});
  
  // Settings stored separately in localStorage
  const [settingsStore, setSettingsStore] = React.useState<AnalysisSettingsStore>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('analysis-settings');
      if (saved) {
        try {
          return JSON.parse(saved);
        } catch (e) {
          console.error('Failed to parse saved analysis settings:', e);
        }
      }
    }
    return {};
  });

  // Save only settings to localStorage (small data)
  React.useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        localStorage.setItem('analysis-settings', JSON.stringify(settingsStore));
      } catch (e) {
        console.error('Failed to save settings to localStorage:', e);
        // Clear old data if quota is exceeded
        localStorage.removeItem('analysis-settings');
      }
    }
  }, [settingsStore]);

  return (
    <AnalysisStoreContext.Provider value={{ 
      imagesStore, 
      setImagesStore, 
      settingsStore, 
      setSettingsStore 
    }}>
      {children}
    </AnalysisStoreContext.Provider>
  );
}

export function useAnalysisStore(projectId?: string | null): CtxValue {
  const ctx = React.useContext(AnalysisStoreContext);
  if (!ctx) {
    throw new Error("useAnalysisStore must be used inside <AnalysisStoreProvider>");
  }

  const { imagesStore, setImagesStore, settingsStore, setSettingsStore } = ctx;

  // Get current data - use defaults if no project data exists
  const images = projectId ? imagesStore[projectId] || [] : [];
  const settings = projectId ? settingsStore[projectId] : null;
  const confidence = settings?.confidence || [0.8]; 
  const iou = settings?.iou || [0.5]; 

  const setImages: React.Dispatch<React.SetStateAction<DroneImage[]>> = (updater) => {
    if (!projectId) return;
    setImagesStore((prev) => {
      const prevImages = prev[projectId] || [];
      const nextImages =
        typeof updater === "function" ? (updater as any)(prevImages) : updater;
      return { 
        ...prev, 
        [projectId]: nextImages 
      };
    });
  };

  const setConfidence = (newConfidence: number[]) => {
    if (!projectId) return;
    setSettingsStore((prev) => {
      const prevSettings = prev[projectId];
      if (!prevSettings) {
        return { 
          ...prev, 
          [projectId]: {
            confidence: newConfidence,
            iou: [0.5]
          }
        };
      }
      return { 
        ...prev, 
        [projectId]: {
          ...prevSettings,
          confidence: newConfidence
        }
      };
    });
  };

  const setIou = (newIou: number[]) => {
    if (!projectId) return;
    setSettingsStore((prev) => {
      const prevSettings = prev[projectId];
      if (!prevSettings) {
        return { 
          ...prev, 
          [projectId]: {
            confidence: [0.8], 
            iou: newIou
          }
        };
      }
      return { 
        ...prev, 
        [projectId]: {
          ...prevSettings,
          iou: newIou
        }
      };
    });
  };

  return { 
    images, 
    setImages, 
    confidence, 
    setConfidence, 
    iou, 
    setIou 
  };
}