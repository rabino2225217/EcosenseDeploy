import socket from "../../services/socket";
import * as React from "react";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../../components/ui/table";
import { Empty } from "../../components/ui/empty";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "../../components/ui/card";
import { Separator } from "../../components/ui/separator";
import { Button } from "../../components/ui/button";
import { Download, Loader, Radar } from "lucide-react";
import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";
import { useParams } from "react-router-dom";

type LandCoverCounts = {
  name: string;
  geojson_id?: string | null;
  counts: Record<string, number>;
};

type SummaryResponse = {
  project_id: string;
  filters: string[];
  land_covers: LandCoverCounts[];
  recorded_at: string;
};

type SummaryRow = {
  landCover: string;
  species: string;
  count: number;
  date?: string;
};

const API_URL = import.meta.env.VITE_API_URL;

export default function AppSummary() {
  const { id: projectId } = useParams<{ id: string }>();
  const [summary, setSummary] = React.useState<SummaryResponse | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [csvLoading, setCsvLoading] = React.useState(false);
  const [pdfLoading, setPdfLoading] = React.useState(false);
  const [projectName, setProjectName] = React.useState<string>("project");

  //Fetch project name
  React.useEffect(() => {
    if (!projectId) return;

    const fetchProject = async () => {
      try {
        const res = await fetch(`${API_URL}/project/${projectId}`, {
          credentials: "include",
        });
        if (!res.ok) throw new Error("Failed to fetch project");
        const data = await res.json();
        if (data?.data?.name) {
          setProjectName(data.data.name);
        }
      } catch (err) {
        console.error("Error fetching project:", err);
      }
    };

    fetchProject();
  }, [projectId]);

  //Fetch summary
  const fetchSummary = React.useCallback(async () => {
    if (!projectId) return;
    setLoading(true);
    try {
      const res = await fetch(
        `${API_URL}/summary/latest?project_id=${projectId}`
      );
      if (!res.ok) throw new Error("Failed to fetch summary");
      const data = await res.json();
      setSummary(data);
    } catch (err) {
      console.error("Failed to fetch summary:", err);
      setSummary(null);
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  React.useEffect(() => {
    fetchSummary();
  }, [fetchSummary]);

  // listen to socket
  React.useEffect(() => {
    if (!projectId) return;

    socket.on("landcover:update", () => {
      fetchSummary();
    });

    return () => {
      socket.off("landcover:update");
    };
  }, [projectId, fetchSummary]);

  const summaryData: SummaryRow[] = React.useMemo(() => {
    if (!summary || !summary.land_covers || summary.land_covers.length === 0)
      return [];

    const rows: SummaryRow[] = [];
    summary.land_covers.forEach((lc) => {
      const counts = lc.counts || {};
      const date = summary.recorded_at
        ? new Date(summary.recorded_at).toISOString().split("T")[0]
        : "";
      if (Object.keys(counts).length === 0) {
        rows.push({ landCover: lc.name, species: "-", count: 0, date });
      } else {
        Object.entries(counts).forEach(([species, count]) => {
          rows.push({ landCover: lc.name, species, count, date });
        });
      }
    });

    return rows;
  }, [summary]);

  const totalCount = React.useMemo(() => {
    return summaryData.reduce((sum, d) => sum + d.count, 0);
  }, [summaryData]);

  //Handle CSV export
  const handleDownloadCSV = async () => {
    if (!summaryData.length) return;
    setCsvLoading(true);

    const url = `${API_URL}/summary/export/detections/${projectId}?name=${encodeURIComponent(
      projectName
    )}`;

    try {
      const response = await fetch(url, { credentials: "include" });
      if (!response.ok) throw new Error("Failed to download CSV");

      const blob = await response.blob();
      const link = document.createElement("a");
      link.href = window.URL.createObjectURL(blob);
      link.download = `${projectName}_Summary.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (err) {
      console.error("Error downloading CSV:", err);
    } finally {
      setCsvLoading(false);
    }
  };

  //Handle PDF export
  const handleDownloadPDF = async () => {
    if (!summaryData.length || !projectId) return;
    setPdfLoading(true);

    try {
      const response = await fetch(
        `${API_URL}/summary/export/detections/${projectId}?name=${encodeURIComponent(projectName)}`,
        { credentials: "include" }
      );

      if (!response.ok) throw new Error("Failed to fetch detailed detections");

      const csvText = await response.text();
      const cleanedCsv = csvText.replace(/"/g, "");
      const [headerLine, ...rows] = cleanedCsv.split("\n").filter(Boolean);
      const headers = headerLine.split(",").map((h) => h.trim());
      const detectionRows = rows.map((r) => r.split(",").map((c) => c.trim()));

      const doc = new jsPDF("p", "mm", "a4");
      const currentYear = new Date().getFullYear();
      const today = new Date().toLocaleDateString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
      });

      //Load Logos
      const toDataUrl = async (url: string): Promise<string> => {
        const res = await fetch(url, { credentials: "same-origin" });
        if (!res.ok) throw new Error(`Failed to load image: ${url}`);
        const blob = await res.blob();
        return await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(String(reader.result));
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });
      };

      const baseUrl = (import.meta as any).env?.BASE_URL || "/";
      const denrSources = [
        `${baseUrl}denr_logo.png`,
        `${baseUrl}denr-logo.png`,
      ];
      const bagongSources = [`${baseUrl}bagong_pilipinas_logo.png`];

      const loadFirstAvailableDataUrl = async (
        sources: string[]
      ): Promise<string> => {
        for (const src of sources) {
          try {
            return await toDataUrl(src);
          } catch (_) {}
        }
        throw new Error("No image sources loaded");
      };

      try {
        const [denrDataUrl, bagongDataUrl] = await Promise.all([
          loadFirstAvailableDataUrl(denrSources),
          loadFirstAvailableDataUrl(bagongSources),
        ]);

        const imgFromDataUrl = (dataUrl: string) =>
          new Promise<HTMLImageElement>((resolve) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.src = dataUrl;
          });

        const [denrImg, bagongImg] = await Promise.all([
          imgFromDataUrl(denrDataUrl),
          imgFromDataUrl(bagongDataUrl),
        ]);

        const denrLogoWidth = 26;
        const bagongLogoWidth = 30;
        const denrHeight =
          (denrImg.naturalHeight / denrImg.naturalWidth) * denrLogoWidth;
        const bagongHeight =
          (bagongImg.naturalHeight / bagongImg.naturalWidth) * bagongLogoWidth;

        doc.addImage(denrDataUrl, "PNG", 18, 16, denrLogoWidth, denrHeight);
        doc.addImage(
          bagongDataUrl,
          "PNG",
          162,
          14,
          bagongLogoWidth,
          bagongHeight
        );
      } catch (e) {
        console.warn("PDF logos failed to load:", e);
      }

      //HEADER
      doc.setFont("helvetica", "bold");
      doc.setFontSize(10);
      doc.text("REPUBLIC OF THE PHILIPPINES", 105, 22, { align: "center" });

      doc.setFontSize(13);
      doc.setTextColor(0, 51, 0);
      doc.text("Department of Environment and Natural Resources", 105, 28, {
        align: "center",
      });

      doc.setTextColor(0, 0, 0);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);
      doc.text("Cordillera Administrative Region", 105, 35, {
        align: "center",
      });
      doc.text("DENR Compound, Gibraltar, Baguio City", 105, 40, {
        align: "center",
      });

      doc.setLineWidth(0.3);
      doc.line(20, 45, 190, 45);

      //TITLE
      doc.setFont("helvetica", "bold");
      doc.setFontSize(16);
      doc.text("DETECTION SUMMARY REPORT", 105, 56, { align: "center" });

      doc.setFont("helvetica", "normal");
      doc.setFontSize(11);
      doc.text(`CY ${currentYear}`, 105, 63, { align: "center" });

      //SUMMARY TABLE
      autoTable(doc, {
        startY: 72,
        head: [["Land Cover", "Species", "Count", "Date"]],
        body: summaryData.map((d) => [
          d.landCover,
          d.species,
          d.count.toString(),
          d.date ?? "",
        ]),
        foot: [["TOTAL", "", totalCount.toString(), ""]],
        theme: "grid",
        styles: {
          font: "helvetica",
          fontSize: 10,
          halign: "center",
          valign: "middle",
          textColor: [30, 30, 30],
          lineColor: [180, 180, 180],
          lineWidth: 0.1,
          cellPadding: 3,
        },
        headStyles: {
          fillColor: [0, 51, 0],
          textColor: 255,
          fontStyle: "bold",
          halign: "center",
          valign: "middle",
          fontSize: 11,
        },
        bodyStyles: {
          fillColor: [255, 255, 255],
          cellPadding: 2.5,
        },
        alternateRowStyles: { fillColor: [248, 248, 248] },
        footStyles: {
          fillColor: [220, 230, 220],
          textColor: [0, 0, 0],
          fontStyle: "bold",
          fontSize: 11,
          cellPadding: 2.5,
        },
        columnStyles: {
          0: { halign: "left" },
          1: { halign: "left" },
          2: { halign: "center" },
          3: { halign: "center" },
        },
      });

      //FOOTER
      const pageHeight = doc.internal.pageSize.height;
      doc.setTextColor(80);
      doc.setFontSize(8);
      doc.text(`Generated by EcoSense on ${today}`, 105, pageHeight - 6, {
        align: "center",
      });

      //DETAILED DETECTIONS PAGE
      doc.addPage();
      doc.setFont("helvetica", "bold");
      doc.setFontSize(14);
      doc.text("DETAILED DETECTIONS", 105, 18, { align: "center" });

      autoTable(doc, {
        startY: 26,
        head: [headers],
        body: detectionRows,
        theme: "grid",
        styles: {
          font: "helvetica",
          fontSize: 8,
          halign: "center",
          valign: "middle",
          textColor: [30, 30, 30],
          lineColor: [200, 200, 200],
          lineWidth: 0.1,
          cellPadding: 1.5,
        },
        headStyles: {
          fillColor: [0, 51, 0],
          textColor: 255,
          fontStyle: "bold",
          fontSize: 9,
        },
        alternateRowStyles: { fillColor: [250, 250, 250] },
        columnStyles: {
          0: { halign: "left" },
          1: { halign: "left" },
          2: { halign: "center" },
          3: { halign: "center" },
          4: { halign: "center" },
        },
        didDrawPage: () => {
          const pageHeight = doc.internal.pageSize.height;
          doc.setFontSize(8);
          doc.setTextColor(100);
          doc.text(`Generated by EcoSense on ${today}`, 105, pageHeight - 6, {
            align: "center",
          });
        },
      });

      //SAVE FILE
      const safeName = projectName
        .replace(/[^\w\s-]/g, "")
        .replace(/\s+/g, "_");
      doc.save(`${safeName}_Summary.pdf`);
    } catch (err) {
      console.error("Error generating PDF:", err);
    } finally {
      setPdfLoading(false);
    }
  };

  return (
    <div className="p-4 sm:p-6 space-y-6 select-none">
      <Card
        className="shadow-md border border-gray-200/50 dark:border-gray-700 dark:bg-[#121212]"
        data-qg="summary-table"
      >
        <CardHeader className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-3 text-center sm:text-left min-w-0">
          <CardTitle className="text-xl font-bold text-gray-800 dark:text-gray-100 truncate">
            Detection Summary
          </CardTitle>
          <div className="flex flex-wrap gap-2 justify-center sm:justify-end">
            <Button
              onClick={handleDownloadCSV}
              variant="outline"
              className="flex items-center gap-2 cursor-pointer dark:border-gray-600 dark:bg-[#181818] dark:text-gray-100"
              disabled={
                summaryData.length === 0 || totalCount === 0 || csvLoading
              }
              data-qg="export-csv"
            >
              {csvLoading ? (
                <>
                  <Loader className="w-4 h-4 animate-spin" /> Exporting Data...
                </>
              ) : (
                <>
                  <Download className="w-4 h-4" /> Export Data (CSV)
                </>
              )}
            </Button>

            <Button
              onClick={handleDownloadPDF}
              variant="outline"
              className="flex items-center gap-2 cursor-pointer dark:border-gray-600 dark:bg-[#181818] dark:text-gray-100"
              disabled={
                summaryData.length === 0 || totalCount === 0 || pdfLoading
              }
              data-qg="export-pdf"
            >
              {pdfLoading ? (
                <>
                  <Loader className="w-4 h-4 animate-spin" /> Exporting
                  Report...
                </>
              ) : (
                <>
                  <Download className="w-4 h-4" /> Export Report (PDF)
                </>
              )}
            </Button>
          </div>
        </CardHeader>

        <Separator className="dark:bg-gray-700/60" />

        <CardContent className="pt-4">
          {loading ? (
            <div className="flex items-center justify-center py-6 text-gray-500 dark:text-gray-300">
              <span className="mr-2">Loading summary</span>
              <Loader className="w-4 h-4 animate-spin text-[#4c8050]" />
            </div>
          ) : summaryData.length === 0 ? (
            <Empty className="space-y-1 text-center">
              <Radar className="w-9 h-9 text-gray-400 mb-1 mx-auto" />
              <h4 className="text-base font-medium text-gray-700 dark:text-gray-100 leading-tight">
                No Summary Yet
              </h4>
              <p className="text-sm text-gray-500 dark:text-gray-400 leading-snug">
                Summary of the detection results will appear here once
                available.
              </p>
            </Empty>
          ) : (
            <div className="overflow-x-auto">
              <Table className="min-w-full text-sm">
                <TableCaption className="text-gray-500 dark:text-gray-400">
                  Summary of detected trees and crops.
                </TableCaption>
                <TableHeader>
                  <TableRow className="bg-gray-50 dark:bg-[#181818]">
                    <TableHead className="text-gray-700 dark:text-gray-100">
                      Land Cover
                    </TableHead>
                    <TableHead className="text-gray-700 dark:text-gray-100">
                      Species
                    </TableHead>
                    <TableHead className="text-center text-gray-700 dark:text-gray-100">
                      Count
                    </TableHead>
                    <TableHead className="text-gray-700 dark:text-gray-100">
                      Date
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {summaryData.map((d, i) => (
                    <TableRow
                      key={i}
                      className="border-gray-200/50 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-[#1e1e1e]"
                    >
                      <TableCell className="font-medium text-gray-800 dark:text-gray-100">
                        {d.landCover}
                      </TableCell>
                      <TableCell className="capitalize text-gray-700 dark:text-gray-200">
                        {d.species}
                      </TableCell>
                      <TableCell className="text-center text-[#4c8050] font-semibold">
                        {d.count}
                      </TableCell>
                      <TableCell className="text-gray-700 dark:text-gray-200">
                        {d.date}
                      </TableCell>
                    </TableRow>
                  ))}
                  <TableRow className="bg-gray-50 dark:bg-[#181818] font-bold">
                    <TableCell className="text-gray-800 dark:text-gray-100">
                      TOTAL
                    </TableCell>
                    <TableCell></TableCell>
                    <TableCell className="text-center text-[#4c8050]">
                      {totalCount}
                    </TableCell>
                    <TableCell></TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
