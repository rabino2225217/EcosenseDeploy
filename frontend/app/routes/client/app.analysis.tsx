import * as React from "react";
import { Button } from "../../components/ui/button";
import { Separator } from "../../components/ui/separator";
import { toast } from "sonner";
import { ZoomIn, ZoomOut, RefreshCcw, Download, Settings } from "lucide-react";
import {
  Trash2,
  Upload,
  Check,
  ChevronDown,
  TreePine,
  Sprout,
  Loader,
  File,
} from "lucide-react";
import { useParams } from "react-router-dom";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../../components/ui/dropdown-menu";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
  type CarouselApi,
} from "../../components/ui/carousel";
import {
  Empty,
  EmptyHeader,
  EmptyTitle,
  EmptyDescription,
} from "../../components/ui/empty";
import { Skeleton } from "../../components/ui/skeleton";
import { useAnalysisStore } from "./analysis-store";
import { Slider } from "../../components/ui/slider";
import { useQuickGuide } from "../../components/client/quick-guide-modal";
type DroneImage = {
  file: File;
  url: string;
  uploadedAt: Date;
  selected: boolean;
  resultUrl?: string;
  detections?: { label: string; coordinates: any; gps_coordinates?: any }[];
};

export default function AnalysisPage() {
  const inputRef = React.useRef<HTMLInputElement | null>(null);
  const thumbnailContainerRef = React.useRef<HTMLDivElement | null>(null);

  const { id: projectId } = useParams<{ id: string }>();
  const { images, setImages } = useAnalysisStore(projectId ?? "");

  const [currentIndex, setCurrentIndex] = React.useState<number>(0);
  const [api, setApi] = React.useState<CarouselApi>();
  const [loading, setLoading] = React.useState(false);
  const [model, setModel] = React.useState<string>("tree");
  const [isImageModalOpen, setIsImageModalOpen] = React.useState(false);
  const [modalImageSrc, setModalImageSrc] = React.useState<string | null>(null);
  const [zoom, setZoom] = React.useState(1);
  const [confidence, setConfidence] = React.useState([0.5]);
  const [iou, setIou] = React.useState([0.5]);

  const API_URL = import.meta.env.VITE_API_URL;

  const openPicker = () => inputRef.current?.click();

  const { setCurrentSection } = useQuickGuide();

  //Handle file input
  const handleFiles = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    if (!files.length) return;
    addFiles(files);

    e.target.value = "";
  };

  //Handle drag & drop
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files ?? []);
    if (!files.length) return;
    addFiles(files);
  };

  //Add files to state (prevent duplicates + limit to 10)
  const addFiles = (files: File[]) => {
    const validFiles = files.filter(
      (f) =>
        f.name.toLowerCase().endsWith(".tif") ||
        f.name.toLowerCase().endsWith(".tiff")
    );

    const invalidCount = files.length - validFiles.length;
    if (invalidCount > 0) {
      toast.error(
        `${invalidCount} file(s) were skipped (only .tif or .tiff allowed)`
      );
    }

    if (validFiles.length === 0) return;

    const existing = new Set(
      images.map((im) => `${im.file.name}-${im.file.size}`)
    );

    const noDuplicates = validFiles.filter(
      (f) => !existing.has(`${f.name}-${f.size}`)
    );

    if (noDuplicates.length < validFiles.length) {
      toast.error(
        "Some files were skipped because they were already uploaded."
      );
    }

    if (noDuplicates.length === 0) return;

    if (images.length >= 10) {
      toast.error("You can only upload up to 10 images.");
      return;
    }

    const availableSlots = 10 - images.length;

    const filesToAdd = noDuplicates.slice(0, availableSlots);

    if (filesToAdd.length < noDuplicates.length) {
      toast.error("Upload limit reached (10 images max). Extra files skipped.");
    }

    const now = new Date();
    const next = filesToAdd.map((f) => ({
      file: f,
      url: URL.createObjectURL(f),
      uploadedAt: now,
      selected: true,
    }));

    setImages((prev) => [...prev, ...next]);
  };

  //Toggle selection
  const toggleAt = (i: number) =>
    setImages((prev) =>
      prev.map((im, idx) =>
        idx === i ? { ...im, selected: !im.selected } : im
      )
    );

  const allSelected = images.length > 0 && images.every((im) => im.selected);
  const toggleAll = () =>
    setImages((prev) => prev.map((im) => ({ ...im, selected: !allSelected })));

  //Remove image
  const removeAt = (i: number) => {
    setImages((prev) => {
      const removed = prev[i];
      if (removed) URL.revokeObjectURL(removed.url);
      return prev.filter((_, idx) => idx !== i);
    });
  };

  //Scroll thumbnail into view
  React.useEffect(() => {
    if (thumbnailContainerRef.current) {
      const activeButton = thumbnailContainerRef.current.children[
        currentIndex
      ] as HTMLElement;
      if (activeButton) {
        activeButton.scrollIntoView({
          behavior: "smooth",
          inline: "center",
          block: "nearest",
        });
      }
    }
  }, [currentIndex]);

  //Prevent ui scroll when modal is open
  React.useEffect(() => {
    if (isImageModalOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }

    return () => {
      document.body.style.overflow = "";
    };
  }, [isImageModalOpen]);

  //Handle analyze selected images
  const analyzeImage = async () => {
    const selectedImages = images.filter((im) => im.selected);
    if (!selectedImages.length) return;
    if (!projectId) {
      toast.error("Project ID is missing in the URL.");
      return;
    }

    setLoading(true);

    //notify app that analysis started
    window.dispatchEvent(
      new CustomEvent("analysis:status", { detail: { loading: true } })
    );

    try {
      const results = await Promise.all(
        selectedImages.map(async (img) => {
          const formData = new FormData();
          formData.append("file", img.file);
          formData.append("project_id", projectId ?? "");
          formData.append("model", model);
          formData.append("confidence", confidence[0].toString());
          formData.append("iou", iou[0].toString());

          const response = await fetch(`${API_URL}/analysis/analyze`, {
            method: "POST",
            body: formData,
          });

          if (!response.ok) {
            let message = "Failed to classify image.";
            try {
              const errorData = await response.json();
              message = errorData.error || errorData.message || message;
            } catch {
              const text = await response.text();
              message = text || message;
            }
            throw new Error(message);
          }

          const data = await response.json();

          if (data.message) toast.info(data.message);

          return {
            ...img,
            resultUrl: data.result_image,
            detections: data.detections,
          };
        })
      );

      setImages((prev) =>
        prev.map((im) => results.find((r) => r.file === im.file) || im)
      );

      setCurrentIndex(0);
      toast.success("Images analyzed successfully!");
    } catch (err: any) {
      console.error(err);
      toast.error(
        err?.message || "Failed to classify images. Please try again."
      );
    } finally {
      setLoading(false);
      window.dispatchEvent(
        new CustomEvent("analysis:status", { detail: { loading: false } })
      );
    }
  };

  const uploadsDisabled = loading;
  const processedImages = images.filter((im) => im.resultUrl);
  const current = processedImages[currentIndex];

  //Sync carousel index
  React.useEffect(() => {
    if (!api) return;
    setCurrentIndex(api.selectedScrollSnap());
    api.on("select", () => setCurrentIndex(api.selectedScrollSnap()));
  }, [api]);

  //Models display
  const modelLabel =
    model === "tree" ? "Tree Detection Model" : "Crop Detection Model";
  const modelIcon =
    model === "tree" ? (
      <TreePine className="h-4 w-4 text-green-600" />
    ) : (
      <Sprout className="h-4 w-4 text-lime-600" />
    );

  return (
    <div className="w-full h-[calc(100vh-5rem)] flex flex-col px-6 pb-6 overflow-hidden select-none">
      {/* HEADER */}
      <div className="flex items-center justify-between mb-4 flex-shrink-0">
        <h1 className="text-2xl font-semibold">Land Analysis</h1>

        <div className="flex items-center gap-2">
          {/* Model Selection Dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild disabled={loading}>
              <Button
                variant="outline"
                className="flex items-center gap-2 cursor-pointer focus:ring-0 focus:ring-offset-0 focus-visible:ring-0"
                data-qg="model-selection"
              >
                {modelIcon}
                {modelLabel}
                <ChevronDown className="h-4 w-4 ml-1" />
              </Button>
            </DropdownMenuTrigger>

            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuItem
                onClick={() => setModel("tree")}
                className={`
              cursor-pointer
              flex items-center
              text-gray-800 dark:text-gray-200
              ${
                model === "tree"
                  ? "bg-green-100 dark:bg-green-900/40 font-medium"
                  : "hover:bg-gray-100 dark:hover:bg-gray-800"
              }
            `}
              >
                <TreePine className="h-4 w-4 mr-2 text-green-600" />
                Tree Detection Model
              </DropdownMenuItem>

              <DropdownMenuItem
                onClick={() => setModel("crop")}
                className={`
              cursor-pointer
              flex items-center
              text-gray-800 dark:text-gray-200
              ${
                model === "crop"
                  ? "bg-green-100 dark:bg-green-900/40 font-medium"
                  : "hover:bg-gray-100 dark:hover:bg-gray-800"
              }
            `}
              >
                <Sprout className="h-4 w-4 mr-2 text-lime-600" />
                Crop Detection Model
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Detection Settings Dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild disabled={loading}>
              <Button
                data-qg="detection-settings"
                variant="outline"
                className="flex items-center gap-2 cursor-pointer focus:ring-0 focus:ring-offset-0 focus-visible:ring-0"
              >
                <Settings className="h-4 w-4" />
                Detection Settings
                <ChevronDown className="h-4 w-4 ml-1" />
              </Button>
            </DropdownMenuTrigger>

            <DropdownMenuContent align="end" className="w-80 p-4">
              <div className="space-y-4">
                <h3 className="font-semibold text-sm text-gray-900 dark:text-gray-100">
                  Detection Parameters
                </h3>

                {/* Confidence Slider */}
                <div
                  className={loading ? "opacity-50 pointer-events-none" : ""}
                >
                  <label className="text-xs text-gray-700 dark:text-gray-300 block mb-2">
                    Confidence: {confidence[0].toFixed(2)}
                  </label>
                  <Slider
                    value={confidence}
                    onValueChange={setConfidence}
                    min={0}
                    max={1}
                    step={0.01}
                    disabled={loading || images.length === 0}
                    className="w-full"
                  />
                </div>

                {/* IoU Slider */}
                <div
                  className={loading ? "opacity-50 pointer-events-none" : ""}
                >
                  <label className="text-xs text-gray-700 dark:text-gray-300 block mb-2">
                    IoU Threshold: {iou[0].toFixed(2)}
                  </label>
                  <Slider
                    value={iou}
                    onValueChange={setIou}
                    min={0}
                    max={1}
                    step={0.01}
                    disabled={loading || images.length === 0}
                    className="w-full"
                  />
                </div>

                <div className="text-xs text-gray-500 dark:text-gray-400 pt-2 border-t border-gray-200 dark:border-gray-700">
                  <p>
                    Adjust confidence and IoU thresholds to control detection
                    sensitivity.
                  </p>
                </div>
              </div>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* MAIN */}
      <div className="flex flex-col lg:flex-row gap-6 flex-1 overflow-hidden">
        {/* LEFT PANEL */}
        <div
          className="
          
          w-full lg:w-1/4 border rounded-2xl p-4 
          bg-white dark:bg-[#1c1c1c] 
          border-gray-200 dark:border-gray-700 
          shadow-sm flex flex-col overflow-hidden
        "
        >
          <div className="flex items-center justify-between mb-4 flex-shrink-0">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Upload & Preview
            </h2>
            {images.length > 0 && (
              <button
                onClick={!loading ? toggleAll : undefined}
                disabled={loading}
                className={`text-xs underline transition-colors cursor-pointer ${
                  loading
                    ? "text-gray-400 dark:text-gray-600 cursor-not-allowed"
                    : "text-gray-500 dark:text-gray-400 hover:text-primary"
                }`}
              >
                {allSelected ? "Unselect All" : "Select All"}
              </button>
            )}
          </div>

          {/* Upload Box */}
          <div
            onDrop={!uploadsDisabled ? handleDrop : undefined}
            onDragOver={(e) => !uploadsDisabled && e.preventDefault()}
            onClick={() => {
              if (!uploadsDisabled) openPicker();
            }}
            className={`
              border-2 border-dashed 
              border-gray-300 dark:border-gray-600 
              rounded-xl h-30 flex flex-col items-center justify-center 
              text-gray-500 dark:text-gray-400 
              transition flex-shrink-0
              ${
                uploadsDisabled
                  ? "opacity-50 cursor-not-allowed pointer-events-none"
                  : "cursor-pointer hover:border-blue-400 dark:hover:border-gray-300"
              }
            `}
            data-qg="upload-preview"
          >
            <Upload className="h-6 w-6 mb-2" />
            <span className="text-sm">Drag & drop or click to upload</span>
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              multiple
              className="hidden"
              onChange={handleFiles}
              disabled={uploadsDisabled}
            />
          </div>

          {/* Uploaded List */}
          <div
            className={`mt-4 overflow-y-auto flex-1 relative custom-scrollbar transition-opacity duration-300 ${
              loading ? "opacity-50 pointer-events-none" : "opacity-100"
            }`}
          >
            {images.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-gray-500 dark:text-gray-400">
                <p className="text-sm">No images uploaded yet</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-2">
                {images.map((img, idx) => (
                  <div
                    key={idx}
                    className={`relative flex items-center gap-3 border rounded-lg p-2.5 shadow-sm transition hover:shadow-md cursor-pointer
                      ${
                        img.selected
                          ? "border-gray-400 bg-gray-50 dark:border-gray-600 dark:bg-[#2b2b2b]"
                          : "border-gray-200 bg-white dark:border-gray-700 dark:bg-[#1f1f1f]"
                      }
                      hover:bg-gray-100 dark:hover:bg-[#2a2a2a]
                    `}
                    onClick={() => {
                      if (!loading) toggleAt(idx);
                    }}
                  >
                    {/* File Icon */}
                    <div className="relative w-12 h-12 flex-shrink-0 flex items-center justify-center rounded-md border border-gray-200 dark:border-gray-700 bg-gray-100 dark:bg-[#2d2d2d]">
                      <File className="w-6 h-6 text-gray-600 dark:text-gray-300" />
                      {img.selected && (
                        <div className="absolute inset-0 flex items-center justify-center bg-black/30 rounded-md">
                          <Check className="h-4 w-4 text-white" />
                        </div>
                      )}
                    </div>

                    <div className="flex flex-col flex-1 overflow-hidden">
                      <p className="text-sm font-medium text-gray-800 dark:text-gray-100 truncate">
                        {img.file.name}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
                        {(img.file.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (!loading) removeAt(idx);
                      }}
                      className="p-1.5 rounded-full hover:bg-gray-100 dark:hover:bg-[#2d2d2d] transition cursor-pointer"
                      title="Remove"
                    >
                      <Trash2 className="h-4 w-4 text-gray-500 dark:text-gray-400 hover:text-red-500 dark:hover:text-red-400" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Analyze Button */}
          <div className="flex justify-end w-full md:w-auto md:mt-10">
            <Button
              onClick={analyzeImage}
              data-qg="analyze"
              className="
                  h-10 px-5 rounded-lg cursor-pointer
                  bg-green-100 text-green-700
                  hover:bg-gray-100
                  dark:bg-green-900/40 dark:text-green-300
                  dark:hover:bg-gray-800
                  border border-green-400 dark:border-green-700
                  disabled:opacity-50 disabled:cursor-not-allowed
                "
              disabled={!images.some((im) => im.selected) || loading}
            >
              {loading ? (
                <>
                  Analyzing
                  <Loader className="h-3 w-3 animate-spin" />
                </>
              ) : (
                "Analyze"
              )}
            </Button>
          </div>
        </div>

        {/* RIGHT PANEL */}
        <div
          className="
          flex-1 border rounded-2xl p-4 
          bg-white dark:bg-[#1c1c1c] 
          border-gray-200 dark:border-gray-700 
          shadow-sm flex flex-col overflow-y-auto
          max-h-[calc(100vh-8rem)]
          custom-scrollbar
        "
        >
          <h2 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100 flex-shrink-0">
            Processed Image
          </h2>

          {/* Skeleton Loading State */}
          {loading ? (
            <div className="flex flex-col gap-3" data-qg="processed-image">
              <Skeleton className="h-[400px] w-full rounded-xl" />
              <div className="flex gap-2">
                <Skeleton className="h-6 w-6 rounded-full" />
                <Skeleton className="h-6 w-6 rounded-full" />
                <Skeleton className="h-6 w-6 rounded-full" />
              </div>
              <div className="mt-4 space-y-2">
                <Skeleton className="h-5 w-1/3" />
                <Skeleton className="h-5 w-full" />
                <Skeleton className="h-5 w-full" />
                <Skeleton className="h-5 w-full" />
              </div>
            </div>
          ) : processedImages.length > 0 ? (
            <>
              {/* Carousel */}
              <div className="relative mb-2 flex-shrink-0">
                <Carousel
                  setApi={setApi}
                  className="w-full"
                  opts={{
                    align: "center",
                    loop: true,
                    startIndex: currentIndex,
                  }}
                >
                  <CarouselContent>
                    {processedImages.map((img, idx) => (
                      <CarouselItem key={idx} className="flex justify-center">
                        <div
                          onClick={() => {
                            setModalImageSrc(img.resultUrl || "");
                            setZoom(1);
                            setIsImageModalOpen(true);
                          }}
                          data-qg="processed-image"
                          className="
                            relative w-full rounded-xl overflow-hidden 
                            border border-gray-200 dark:border-gray-700
                            bg-white dark:bg-[#101010]
                            flex items-center justify-center h-[400px]
                            cursor-zoom-in
                          "
                        >
                          <img
                            src={img.resultUrl}
                            alt={`Processed ${idx + 1}`}
                            loading="lazy"
                            className="max-h-full max-w-full object-contain"
                          />
                        </div>
                      </CarouselItem>
                    ))}
                  </CarouselContent>

                  {processedImages.length > 1 && (
                    <>
                      <CarouselPrevious className="left-2 top-1/2 -translate-y-1/2 bg-white/80 hover:bg-white shadow-lg rounded-full transition-transform duration-200 hover:scale-110 active:scale-95" />
                      <CarouselNext className="right-2 top-1/2 -translate-y-1/2 bg-white/80 hover:bg-white shadow-lg rounded-full transition-transform duration-200 hover:scale-110 active:scale-95" />
                    </>
                  )}
                </Carousel>
              </div>

              {/* Thumbnails */}
              <div
                ref={thumbnailContainerRef}
                className="flex gap-2 overflow-x-auto pt-1 pb-3 pl-2 flex-shrink-0 max-w-full custom-scrollbar"
              >
                {processedImages.map((_, idx) => (
                  <button
                    key={idx}
                    onClick={() => api?.scrollTo(idx)}
                    className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-semibold border transition-all duration-200 cursor-pointer
                      ${
                        idx === currentIndex
                          ? "text-[#4c8050] ring-4"
                          : "bg-gray-100 border-gray-300 text-gray-600 hover:bg-gray-200"
                      }`}
                    style={
                      idx === currentIndex
                        ? {
                            backgroundColor: "#e9f3ea",
                            borderColor: "#4c8050",
                            boxShadow: "0 0 0 3px rgba(76, 128, 80, 0.2)",
                          }
                        : {}
                    }
                  >
                    {idx + 1}
                  </button>
                ))}
              </div>

              <Separator className="my-1 flex-shrink-0" />

              {/* Detection Breakdown */}
              {current?.detections && current.detections.length > 0 ? (
                <div
                  className="
                    mt-4 
                    border rounded-xl p-4 shadow-sm
                    bg-white dark:bg-[#1c1c1c]
                    border-gray-200 dark:border-gray-700
                  "
                >
                  <h3 className="font-semibold mb-3 text-gray-800 dark:text-gray-100 flex items-center gap-2">
                    <span className="inline-block w-2 h-2 rounded-full bg-[#4c8050]" />
                    Detection Breakdown
                  </h3>

                  {(() => {
                    const counts: Record<string, number> = {};
                    current.detections.forEach((d) => {
                      counts[d.label] = (counts[d.label] || 0) + 1;
                    });

                    return (
                      <div className="space-y-2 text-sm">
                        {Object.entries(counts).map(([label, count], i) => (
                          <div
                            key={i}
                            className="
                              flex justify-between items-center 
                              bg-white dark:bg-[#252525]
                              border border-gray-200 dark:border-gray-700
                              rounded-lg px-3 py-2
                              hover:shadow-sm transition
                            "
                          >
                            <span className="text-gray-700 dark:text-gray-100 capitalize">
                              {label}
                            </span>
                            <span className="font-semibold text-[#4c8050]">
                              {count}
                            </span>
                          </div>
                        ))}

                        <div
                          className="
                            flex justify-between items-center 
                            border-t border-gray-200 dark:border-gray-700
                            pt-3 mt-3 font-semibold 
                            text-gray-800 dark:text-gray-100
                          "
                        >
                          <span>Total Detections</span>
                          <span style={{ color: "#4c8050" }}>
                            {current.detections.length}
                          </span>
                        </div>
                      </div>
                    );
                  })()}
                </div>
              ) : (
                <div className="text-gray-400 text-sm italic mt-2">
                  No detections found.
                </div>
              )}
            </>
          ) : (
            <div
              className="flex-1 flex items-center justify-center"
              data-qg="processed-image"
            >
              <Empty className="p-6">
                <EmptyHeader>
                  <EmptyTitle>No Processed Image Yet</EmptyTitle>
                  <EmptyDescription>
                    Upload a GeoTIFF image (.tif or .tiff) and analyze it to
                    view detection results here
                  </EmptyDescription>
                </EmptyHeader>
              </Empty>
            </div>
          )}
        </div>
      </div>

      {/* Processed image modal */}
      {isImageModalOpen && (
        <div
          className="fixed inset-0 z-[9999] bg-black/70 backdrop-blur-sm flex items-center justify-center p-4"
          onClick={() => setIsImageModalOpen(false)}
        >
          <div
            className="relative bg-[#0f0f0f] dark:bg-[#0f0f0f] rounded-xl overflow-hidden max-w-6xl w-full max-h-[90vh] flex items-center justify-center"
            onClick={(e) => e.stopPropagation()}
          >
            {/* close */}
            <button
              onClick={() => setIsImageModalOpen(false)}
              className="absolute top-3 right-3 h-9 w-9 rounded-full bg-black/70 text-white flex items-center justify-center hover:bg-black cursor-pointer z-20"
              aria-label="Close"
            >
              ✕
            </button>

            {/* zoom controls */}
            <div className="absolute top-3 right-16 flex gap-2 z-20">
              <button
                onClick={() => setZoom((z) => Math.max(z - 0.25, 0.5))}
                className="h-9 w-9 rounded-full bg-black/60 text-white flex items-center justify-center hover:bg-black cursor-pointer"
                aria-label="Zoom out"
              >
                <ZoomOut className="h-4 w-4" />
              </button>
              <button
                onClick={() => setZoom((z) => Math.min(z + 0.25, 4))}
                className="h-9 w-9 rounded-full bg-black/60 text-white flex items-center justify-center hover:bg-black cursor-pointer"
                aria-label="Zoom in"
              >
                <ZoomIn className="h-4 w-4" />
              </button>
              <button
                onClick={() => {
                  if (!modalImageSrc) return;
                  const link = document.createElement("a");
                  link.href = modalImageSrc;
                  link.download = "image.png";
                  link.click();
                }}
                className="h-9 w-9 rounded-full bg-black/60 text-white flex items-center justify-center hover:bg-black cursor-pointer"
                aria-label="Download image"
              >
                <Download className="h-4 w-4" />
              </button>
            </div>

            {/* draggable / zoomable image */}
            <DragZoomImage
              src={modalImageSrc}
              zoom={zoom}
              onPanReset={() => {}}
            />
          </div>
        </div>
      )}
    </div>
  );
}

//Processed image drag zoom function
function DragZoomImage({
  src,
  zoom,
  onPanReset,
}: {
  src: string | null;
  zoom: number;
  onPanReset: () => void;
}) {
  const imgRef = React.useRef<HTMLImageElement | null>(null);
  const [isPanning, setIsPanning] = React.useState(false);
  const [currentZoom, setCurrentZoom] = React.useState(zoom);

  const panRef = React.useRef({ x: 0, y: 0 });
  const startRef = React.useRef({ x: 0, y: 0 });

  React.useEffect(() => {
    setCurrentZoom(zoom);
  }, [zoom]);

  React.useEffect(() => {
    if (currentZoom === 1) {
      panRef.current = { x: 0, y: 0 };
      if (imgRef.current) imgRef.current.style.transform = "none";
      onPanReset();
    } else {
      const { x, y } = panRef.current;
      if (imgRef.current) {
        imgRef.current.style.transform = `translate(${x}px, ${y}px) scale(${currentZoom})`;
      }
    }
  }, [currentZoom, onPanReset]);

  if (!src) return null;

  const applyTransform = (x: number, y: number, z: number) => {
    if (imgRef.current) {
      imgRef.current.style.transform = `translate(${x}px, ${y}px) scale(${z})`;
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsPanning(true);
    startRef.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isPanning) return;
    const dx = e.clientX - startRef.current.x;
    const dy = e.clientY - startRef.current.y;

    const nextX = panRef.current.x + dx;
    const nextY = panRef.current.y + dy;

    applyTransform(nextX, nextY, currentZoom);
  };

  const handleMouseUp = () => {
    if (!isPanning) return;
    setIsPanning(false);

    if (imgRef.current) {
      const style = imgRef.current.style.transform;
      const match = style.match(/translate\(([-\d.]+)px,\s*([-\d.]+)px\)/);
      if (match) {
        panRef.current = {
          x: parseFloat(match[1]),
          y: parseFloat(match[2]),
        };
      }
    }
  };

  //Handle mouse wheel zoom
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.1 : -0.1;
    const newZoom = Math.min(4, Math.max(0.5, currentZoom + delta));

    setCurrentZoom(newZoom);

    const { x, y } = panRef.current;
    applyTransform(x, y, newZoom);
  };

  return (
    <div
      className={`max-h-[85vh] max-w-full overflow-hidden ${
        isPanning ? "cursor-grabbing" : "cursor-grab"
      }`}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseUp}
      onMouseUp={handleMouseUp}
      onWheel={handleWheel}
    >
      <img
        ref={imgRef}
        src={src}
        alt="Processed image"
        className="max-h-[85vh] max-w-full object-contain select-none"
        draggable={false}
      />
    </div>
  );
}
