import { useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "../ui/dialog";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Upload, TreePine, Sprout, Loader } from "lucide-react";
import { toast } from "sonner";

interface Props {
  open: boolean;
  onClose: () => void;
  onAdd: (data: { name: string; type: string; file: File }) => Promise<{ success: boolean; message: string }>;
}

export default function AddLandCoverModal({ open, onClose, onAdd }: Props) {
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [type, setType] = useState("");
  const [loading, setLoading] = useState(false);
  const MAX_LENGTH = 30;

  useEffect(() => {
    if (open) {
      setName("");
      setFile(null);
      setType("");
      setLoading(false);
    }
  }, [open]);

  const handleAdd = async () => {
    if (!name.trim() || !file || !type) {
      toast.error("Please complete all fields before submitting.");
      return;
    }

    if (name.trim().length > MAX_LENGTH) {
      toast.error(`Land cover name cannot exceed ${MAX_LENGTH} characters.`);
      return;
    }

    try {
      setLoading(true);
      const result = await onAdd({ name: name.trim(), type, file });

      if (result.success) {
        toast.success(result.message);
        onClose();
      } else {
        toast.error(result.message);
      }
    } catch (err: any) {
      console.error(err);
      toast.error(err?.message || "Failed to add land cover.");
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      if (selected.size > 15 * 1024 * 1024) {
        toast.error("File size exceeds 15 MB limit.");
        setFile(null);
      } else if (!selected.name.toLowerCase().endsWith(".tif")) {
        toast.error("Only GeoTIFF (.tif) files are allowed.");
        setFile(null);
      } else {
        setFile(selected);
        toast.success(`"${selected.name}" selected.`);
      }
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-md p-6 rounded-2xl border border-gray-200 shadow-lg">
        <DialogHeader>
          <DialogTitle className="text-xl font-semibold text-gray-800">
            Add New Land Cover
          </DialogTitle>
          <DialogDescription className="text-[13px] text-gray-500 leading-tight">
            Upload a GeoTIFF raster and specify its type (Trees or Crops).
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-5 mt-3">
          {/* Land Cover Name */}
          <div className="space-y-2">
            <Label className="text-gray-700 font-medium">Land Cover Name</Label>
            <Input
              placeholder="Enter land cover name..."
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={loading}
              maxLength={MAX_LENGTH}
            />
            <p className="text-sm text-gray-500 text-right">
              {name.length}/{MAX_LENGTH}
            </p>
          </div>

          {/* File Upload */}
          <div className="space-y-2">
            <Label className="text-gray-700 font-medium">
              Upload Raster File (GeoTIFF)
            </Label>
            <label
              htmlFor="raster-upload"
              className={`flex flex-col items-center justify-center rounded-lg border border-dashed border-gray-300 bg-gray-50 py-8 transition-colors ${
                loading
                  ? "opacity-50 cursor-not-allowed"
                  : "hover:bg-gray-100 cursor-pointer"
              }`}
            >
              <Upload className="w-6 h-6 text-gray-500 mb-2" />
              <span className="text-sm text-gray-600">
                {file ? <b>{file.name}</b> : "Click to upload .tif file"}
              </span>
              <Input
                id="raster-upload"
                type="file"
                accept=".tif,.tiff"
                className="hidden"
                onChange={handleFileChange}
                disabled={loading}
              />
            </label>
            <p className="text-xs text-gray-500 text-center">
              Maximum file size: <b>15 MB</b>
            </p>
          </div>

          {/* Dropdown for Type */}
          <div className="space-y-2">
            <Label className="text-gray-700 font-medium">Land Cover Type</Label>
            <Select value={type} onValueChange={setType} disabled={loading}>
              <SelectTrigger className="cursor-pointer">
                <SelectValue placeholder="Select type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Trees" className="cursor-pointer">
                  <div className="flex items-center gap-2">
                    <TreePine className="h-4 w-4 text-green-600" />
                    <span>Trees</span>
                  </div>
                </SelectItem>
                <SelectItem value="Crops" className="cursor-pointer">
                  <div className="flex items-center gap-2">
                    <Sprout className="h-4 w-4 text-green-600" />
                    <span>Crops</span>
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <DialogFooter className="pt-4">
          <Button
            variant="outline"
            className="border-gray-300 text-gray-700 hover:bg-gray-100 cursor-pointer"
            onClick={onClose}
            disabled={loading}
          >
            Cancel
          </Button>

          <Button
            className="bg-black hover:bg-gray-900 text-white flex items-center gap-2 cursor-pointer"
            onClick={handleAdd}
            disabled={!name.trim() || !file || !type || loading}
          >
            {loading ? (
              <>
                <span>Adding</span>
                <Loader className="h-4 w-4 animate-spin" />
              </>
            ) : (
              "Add Land Cover"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}