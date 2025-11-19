import { useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "../ui/dialog";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Loader } from "lucide-react";
import { toast } from "sonner";

interface EditProjectModalProps {
  open: boolean;
  onClose: () => void;
  project: { id: string; name: string } | null;
  onSave: (newName: string) => Promise<{ success: boolean; message: string }>;
}

export default function EditProjectModal({
  open,
  onClose,
  project,
  onSave,
}: EditProjectModalProps) {
  const [name, setName] = useState(project?.name || "");
  const [loading, setLoading] = useState(false);
  const MAX_LENGTH = 50;

  useEffect(() => {
    if (open && project) {
      setName(project.name); 
    }
  }, [open, project]);

  const handleSave = async () => {
    if (!name.trim()) {
      toast.error("Project name cannot be empty.");
      return;
    }

    if (name.trim().length > MAX_LENGTH) {
      toast.error(`Project name cannot exceed ${MAX_LENGTH} characters.`);
      return;
    }

    try {
      setLoading(true);
      const result = await onSave(name.trim());

      if (result.success) {
        toast.success(result.message || "Project renamed successfully!");
        onClose();
      } else {
        toast.error(result.message || "Failed to rename project.");
      }
    } catch (err: any) {
      console.error("Error renaming project:", err);
      toast.error("Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={!loading ? onClose : () => {}}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Rename Project</DialogTitle>
        </DialogHeader>

        <div className="space-y-2">
          <Input
            value={name}
            onChange={(e) => setName(e.target.value)}
            disabled={loading}
            placeholder="Enter new project name"
            maxLength={MAX_LENGTH}
          />
          <p className="text-sm text-gray-500 text-right">
            {name.length}/{MAX_LENGTH}
          </p>
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={onClose}
            disabled={loading}
            className="cursor-pointer"
          >
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            disabled={loading}
            className="cursor-pointer flex items-center gap-2"
          >
            {loading ? (
              <>
                Saving
                <Loader className="h-4 w-4 animate-spin" />
              </>
            ) : (
              "Save"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}