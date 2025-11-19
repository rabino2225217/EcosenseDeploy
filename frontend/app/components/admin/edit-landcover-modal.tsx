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

interface Props {
  open: boolean;
  onClose: () => void;
  onSave: (newName: string) => Promise<{ success: boolean; message: string }>;
  currentName: string;
}

export default function EditLandCoverModal({
  open,
  onClose,
  onSave,
  currentName,
}: Props) {
  const [name, setName] = useState(currentName);
  const [loading, setLoading] = useState(false);
  const MAX_LENGTH = 30;

  useEffect(() => {
    if (open) setName(currentName);
  }, [open, currentName]);

  const handleSave = async () => {
    if (!name.trim()) {
      toast.error("Land cover name cannot be empty.");
      return;
    }

    if (name.trim().length > MAX_LENGTH) {
      toast.error(`Land cover name cannot exceed ${MAX_LENGTH} characters.`);
      return;
    }

    try {
      setLoading(true);
      const result = await onSave(name.trim());
      if (result.success) {
        toast.success(result.message);
        onClose();
      } else {
        toast.error(result.message);
      }
    } catch (err: any) {
      console.error("Error updating:", err);
      toast.error(err?.message || "Failed to update land cover.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={!loading ? onClose : () => {}}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Edit Land Cover</DialogTitle>
        </DialogHeader>

        <div className="space-y-2">
          <Input
            value={name}
            onChange={(e) => setName(e.target.value)}
            disabled={loading}
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