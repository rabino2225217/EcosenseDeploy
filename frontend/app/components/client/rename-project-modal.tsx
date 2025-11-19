import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "../ui/dialog";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Loader } from "lucide-react";
import { toast } from "sonner";

interface Props {
  open: boolean;
  onClose: () => void;
  onRename: (newName: string) => Promise<{ message: string; success?: boolean }>;
  currentName: string;
}

export default function RenameProjectModal({ open, onClose, onRename, currentName }: Props) {
  const [name, setName] = useState(currentName);
  const [loading, setLoading] = useState(false);
  const maxLength = 50;

  const handleSave = async () => {
    if (!name.trim()) {
      toast.error("Project name cannot be empty.");
      return;
    }

    if (name.length > maxLength) {
      toast.error(`Project name cannot exceed ${maxLength} characters.`);
      return;
    }

    setLoading(true);
    try {
      const result = await onRename(name.trim());
      toast[result.success === false ? "error" : "success"](result.message);
      if (result.success !== false) onClose();
    } catch (err: any) {
      console.error(err);
      toast.error(err.message || "Failed to rename project.");
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

        <div className="space-y-1">
          <Input
            value={name}
            onChange={(e) => setName(e.target.value)}
            maxLength={maxLength}
            disabled={loading}
            placeholder="Enter new project name"
          />
          <p className="text-sm text-muted-foreground text-right">
            {name.length}/{maxLength}
          </p>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose} disabled={loading} className="cursor-pointer">
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={loading} className="cursor-pointer flex items-center gap-2">
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