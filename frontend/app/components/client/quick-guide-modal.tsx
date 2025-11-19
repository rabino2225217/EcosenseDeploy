// In quick-guide-modal.tsx
import * as React from "react";
import { useLocation } from "react-router-dom";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogClose,
} from "../ui/dialog";
import { Button } from "../ui/button";

type TourStep = {
  title: string;
  desc: string;
  selector?: string;
  waitForClick?: boolean;
};

type QuickGuideContextValue = {
  openTour: () => void;
  openInfo: () => void;
  closeAll: () => void;
  currentSection: "projects" | "analysis" | "map" | "summary";
  setCurrentSection: (
    section: "projects" | "analysis" | "map" | "summary"
  ) => void;
};

const QuickGuideContext = React.createContext<QuickGuideContextValue | null>(
  null
);

// move steps OUTSIDE so they're stable
const PROJECTS_TOUR_STEPS: TourStep[] = [
  {
    title: "Welcome to EcoSense",
    desc: "Let's take a quick tour of the EcoSense detection system. Click 'Next' to continue.",
  },
  {
    title: "Getting Started",
    desc: "On the All Projects page, click the New button to create a new project. Enter the details and then open the project.",
    selector: '[data-qg="new-project"]',
  },
  {
    title: "All Projects",
    desc: "This is where the newly created projects will appear. You can view and manage all your projects here.",
    selector: '[data-qg="all-projects-panel"]',
  },
  {
    title: "Sort Projects",
    desc: "You can sort your projects by name or date. Click the Sort button to choose how you want to view your projects.",
    selector: '[data-qg="sort-projects"]',
  },
  {
    title: "Search Projects",
    desc: "Use the search bar to quickly find projects by name. Just type a keyword and the results will appear instantly.",
    selector: '[data-qg="search-projects"]',
  },
  {
    title: "All Done",
    desc: "You're all set! Now create a new project and enter it to start working on your analysis. You can always revisit this guide by clicking on the 'Quick Guide' button.",
  },
];

const ANALYSIS_TOUR_STEPS: TourStep[] = [
  {
    title: "Land Analysis Overview",
    desc: "Welcome to the Land Analysis page! Here you can analyze drone imagery to detect trees or crops. Let's explore the key features.",
  },
  {
    title: "Navigation Sidebar",
    desc: "Use the sidebar to navigate between different sections: All Projects to switch or create projects, Land Analysis for detection, Map View to see geolocated results, and Summary & Downloads to export reports.",
    selector: '[data-qg="app-sidebar"]',
  },
  {
    title: "Model Selection",
    desc: "Choose the appropriate detection model for your analysis. Select Tree Detection Model for trees or Crop Detection Model for crops to ensure accurate results.",
    selector: '[data-qg="model-selection"]',
  },
  {
    title: "Upload & Preview Panel",
    desc: "Upload your drone imagery here. You can upload multiple GeoTIFF images (.tif/.tiff) as long as they're not duplicates. Select images and adjust confidence/IoU settings before analysis.",
    selector: '[data-qg="upload-preview"]',
  },
  {
    title: "Detection Settings",
    desc: "Before analyzing your images, review the detection parameters. Adjust the Confidence Threshold and IoU Threshold based on your preference to control how strict or lenient the model should be. Higher confidence reduces false positives, while adjusting IoU affects how overlapping detections are merged.",
    selector: '[data-qg="detection-settings"]',
  },
  {
    title: "Analyze the Images",
    desc: "Once you're done uploading your images, click the Analyze button to process them and generate detection results.",
    selector: '[data-qg="analyze"]',
  },
  {
    title: "Processed Image Panel",
    desc: "View your analysis results here. After processing, you'll see the detected objects with bounding boxes. Click on the image to zoom and examine details closely.",
    selector: '[data-qg="processed-image"]',
  },
  {
    title: "Analysis Complete",
    desc: "You're ready to analyze! Upload images, select the right model, and examine your results. Use Map View to see real-world locations and Summary to export your findings.",
  },
];

const MAP_TOUR_STEPS: TourStep[] = [
  {
    title: "Map View",
    desc: "Visualize your detections on the map. Select a project layer, such as BLISTT Benguet or Latrinidad, to see where trees or crops were detected in real-world locations.",
    selector: '[data-qg="map-view"]',
  },
  {
    title: "Apply Filters",
    desc: "Refine your map results by selecting or deselecting detected species. You can also adjust the confidence threshold to display only detections above your chosen level.",
    selector: '[data-qg="filters-panel"]',
  },
  {
    title: "Legend & Counts",
    desc: "Check the legend to see counts for each detected species. The legend updates dynamically based on your filters and confidence settings, helping you understand species distribution.",
    selector: '[data-qg="legend-panel"]',
  },
];

const SUMMARY_TOUR_STEPS: TourStep[] = [
  {
    title: "Summary & Downloads",
    desc: "View the detection summary for your project. You can export your results as a CSV file or generate a PDF report for easy sharing and documentation.",
    selector: '[data-qg="summary-downloads"]',
  },
  {
    title: "Detection Summary Table",
    desc: "This table shows all your detection results organized by land cover area and species. You can see counts for each species and the total detections across your project.",
    selector: '[data-qg="summary-table"]',
  },
  {
    title: "Export Data (CSV)",
    desc: "Download your detection data as a CSV file for further analysis in spreadsheet applications like Excel or Google Sheets.",
    selector: '[data-qg="export-csv"]',
  },
  {
    title: "Export Report (PDF)",
    desc: "Generate a professional PDF report with your detection summary and detailed results, formatted for official documentation and presentations.",
    selector: '[data-qg="export-pdf"]',
  },
  {
    title: "Summary Complete",
    desc: "You're all set! Use the summary page to review your detection results and export them for reporting and analysis purposes.",
  },
];

export function QuickGuideProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const location = useLocation();
  const [tourOpen, setTourOpen] = React.useState(false);
  const [tourStep, setTourStep] = React.useState(0);
  const [waitingForClick, setWaitingForClick] = React.useState(false);
  const [currentSection, setCurrentSection] = React.useState<
    "projects" | "analysis" | "map" | "summary"
  >("projects");

  // highlight box
  const [highlight, setHighlight] = React.useState<{
    top: number;
    left: number;
    width: number;
    height: number;
  } | null>(null);

  // static info
  const [infoOpen, setInfoOpen] = React.useState(false);

  // Auto-detect section based on route
  React.useEffect(() => {
    const path = location.pathname;

    // Check if we're on a map page
    if (path.includes("/map")) {
      setCurrentSection("map");
    }
    // Check if we're on an analysis page
    else if (path.includes("/analysis")) {
      setCurrentSection("analysis");
    }
    // Check if we're on a summary page
    else if (path.includes("/summary")) {
      setCurrentSection("summary");
    }
    // Check if we're on the main projects page
    else if (
      path === "/app" ||
      path === "/app/" ||
      (path.includes("/app/projects") &&
        !path.includes("/analysis") &&
        !path.includes("/map") &&
        !path.includes("/summary"))
    ) {
      setCurrentSection("projects");
    }
  }, [location.pathname]);

  // Get current tour steps based on section
  const getCurrentTourSteps = () => {
    switch (currentSection) {
      case "projects":
        return PROJECTS_TOUR_STEPS;
      case "analysis":
        return ANALYSIS_TOUR_STEPS;
      case "map":
        return MAP_TOUR_STEPS;
      case "summary":
        return SUMMARY_TOUR_STEPS;
      default:
        return PROJECTS_TOUR_STEPS;
    }
  };

  const currentTourSteps = getCurrentTourSteps();
  const totalSteps = currentTourSteps.length;

  // Safe current step with bounds checking
  const currentStep = currentTourSteps[tourStep] ||
    currentTourSteps[0] || {
      title: "Quick Guide",
      desc: "Welcome to the quick guide!",
    };

  const openTour = () => {
    setTourStep(0);
    setTourOpen(true);
  };

  const openInfo = () => {
    setInfoOpen(true);
  };

  const closeAll = () => {
    setTourOpen(false);
    setInfoOpen(false);
    setHighlight(null);
    setWaitingForClick(false);
  };

  // effect only depends on tourOpen + tourStep
  React.useEffect(() => {
    if (!tourOpen) {
      setHighlight(null);
      setWaitingForClick(false);
      return;
    }

    // Ensure tourStep is within bounds
    if (tourStep >= totalSteps) {
      setTourStep(totalSteps - 1);
      return;
    }

    const step = currentTourSteps[tourStep];

    if (step?.selector) {
      const el = document.querySelector(step.selector) as HTMLElement | null;

      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });

        const rect = el.getBoundingClientRect();
        setHighlight({
          top: rect.top,
          left: rect.left,
          width: rect.width,
          height: rect.height,
        });

        if (step.waitForClick) {
          setWaitingForClick(true);

          const handleClick = () => {
            setWaitingForClick(false);
            setTourStep((prev) => Math.min(prev + 1, totalSteps - 1));
          };

          el.addEventListener("click", handleClick, { once: true });

          return () => {
            el.removeEventListener("click", handleClick);
          };
        } else {
          setWaitingForClick(false);
        }
      } else {
        // selector not found
        setHighlight(null);
        setWaitingForClick(false);
      }
    } else {
      // no selector on this step
      setHighlight(null);
      setWaitingForClick(false);
    }
  }, [tourOpen, tourStep, totalSteps, currentTourSteps]);

  // Check if this is the last step of the current section
  const isLastStepOfSection = tourStep === totalSteps - 1;

  // Get section label for display
  const getSectionLabel = () => {
    switch (currentSection) {
      case "projects":
        return "Projects Guide";
      case "analysis":
        return "Analysis Guide";
      case "map":
        return "Map Guide";
      default:
        return "Quick Guide";
    }
  };

  return (
    <QuickGuideContext.Provider
      value={{
        openTour,
        openInfo,
        closeAll,
        currentSection,
        setCurrentSection,
      }}
    >
      {children}

      {/* highlight outline - LOWER z-index */}
      {tourOpen && highlight && (
        <>
          <div className="fixed inset-0 pointer-events-none z-[45] bg-transparent" />
          <div
            className="fixed z-[50] rounded-lg ring-2 ring-emerald-400/90 bg-emerald-400/5 pointer-events-none"
            style={{
              top: highlight.top,
              left: highlight.left,
              width: highlight.width,
              height: highlight.height,
            }}
          />
        </>
      )}

      {/* PRODUCT TOUR dialog */}
      <Dialog open={tourOpen} onOpenChange={setTourOpen}>
        <DialogContent className="max-w-md select-none">
          <DialogHeader>
            <div>
              <DialogTitle className="font-semibold">
                {currentStep.title}
              </DialogTitle>
              <div className="mt-1 text-xs text-muted-foreground">
                {getSectionLabel()} - Step {Math.min(tourStep + 1, totalSteps)}{" "}
                of {totalSteps}
              </div>
            </div>
            <DialogDescription className="dark:text-white/90 mt-3">
              {currentStep.desc}
            </DialogDescription>
          </DialogHeader>

          {/* progress bar */}
          <div className="mt-2 h-1.5 w-full rounded-full bg-muted">
            <div
              className="h-1.5 rounded-full bg-emerald-500 transition-all"
              style={{
                width: `${(Math.min(tourStep + 1, totalSteps) / totalSteps) * 100}%`,
              }}
            />
          </div>

          <div className="mt-6 flex items-center justify-between gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setTourStep((s) => Math.max(0, s - 1))}
              disabled={tourStep === 0}
              className="cursor-pointer"
            >
              Previous
            </Button>

            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setTourOpen(false)}
                className="text-muted-foreground hover:text-foreground cursor-pointer"
              >
                Skip tutorial
              </Button>

              {isLastStepOfSection ? (
                <DialogClose asChild>
                  <Button
                    size="sm"
                    className="bg-emerald-500 hover:bg-emerald-600 text-white cursor-pointer"
                    onClick={() => setTourOpen(false)}
                  >
                    Done
                  </Button>
                </DialogClose>
              ) : (
                <Button
                  size="sm"
                  className="bg-emerald-500 hover:bg-emerald-600 text-white cursor-pointer"
                  onClick={() =>
                    setTourStep((s) => Math.min(totalSteps - 1, s + 1))
                  }
                >
                  Next
                </Button>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* INFO dialog */}
      <Dialog open={infoOpen} onOpenChange={setInfoOpen}>
        <DialogContent className="max-w-md select-none">
          <DialogHeader>
            <DialogTitle>Quick Guide - Tree and Crop Detection</DialogTitle>
            <DialogDescription>
              Learn how to use the tree and crop detection system
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 text-sm text-muted-foreground dark:text-white">
            <div className="space-y-1">
              <p className="text-sm font-semibold text-foreground dark:text-emerald-300">
                Getting Started
              </p>
              <p className="dark:text-white">
                Log in to access the{" "}
                <span className="font-medium">All Projects</span> page. Click{" "}
                <span className="font-medium">New Project</span>, enter name,
                location (e.g. BLISTT), and description, then create and open
                the project.
              </p>
            </div>

            <div className="space-y-1">
              <p className="text-sm font-semibold text-foreground dark:text-emerald-300">
                Land Analysis
              </p>
              <ul className="list-disc list-inside space-y-1 dark:text-white">
                <li>Select a model: Tree Detection or Crop Detection.</li>
                <li>Upload your drone imagery.</li>
                <li>
                  Click <span className="font-medium">Analyze</span> to detect
                  and count using bounding boxes.
                </li>
              </ul>
            </div>

            <div className="space-y-1">
              <p className="text-sm font-semibold text-foreground dark:text-emerald-300">
                Map View
              </p>
              <p className="dark:text-white">
                View analyzed imagery on an interactive map to verify that
                detection locations align with real-world coordinates.
              </p>
            </div>

            <div className="space-y-1">
              <p className="text-sm font-semibold text-foreground dark:text-emerald-300">
                Summary &amp; Downloads
              </p>
              <p className="dark:text-white">
                Access summaries, stats, and download reports in
                <span className="font-medium"> PDF</span> or
                <span className="font-medium"> CSV</span> format.
              </p>
            </div>
          </div>

          <div className="flex justify-end pt-2">
            <DialogClose asChild>
              <Button variant="outline" size="sm">
                Close
              </Button>
            </DialogClose>
          </div>
        </DialogContent>
      </Dialog>
    </QuickGuideContext.Provider>
  );
}

export function useQuickGuide() {
  const ctx = React.useContext(QuickGuideContext);
  if (!ctx) {
    throw new Error("useQuickGuide must be used inside <QuickGuideProvider>");
  }
  return ctx;
}
