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
import { toast } from "sonner";
import { Loader } from "lucide-react";

interface Props {
  open: boolean;
  onClose: () => void;
  onReset: (newPassword: string) => Promise<boolean>; 
  userName: string;
}

export default function ResetPasswordModal({
  open,
  onClose,
  onReset,
  userName,
}: Props) {
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (open) {
      setPassword("");
      setConfirmPassword("");
    }
  }, [open]);

  const handleSave = async () => {
    if (!password.trim() || !confirmPassword.trim()) {
      toast.error("Please fill in all fields");
      return;
    }

    if (password !== confirmPassword) {
      toast.error("Passwords do not match");
      return;
    }

    setLoading(true);
    try {
      const success = await onReset(password.trim());
      if (success) {
        toast.success(`Password for ${userName} has been reset successfully!`);
        setPassword("");
        setConfirmPassword("");
        onClose(); 
      } else {
        toast.error("Failed to reset password");
      }
    } catch (err) {
      console.error(err);
      toast.error("An error occurred while resetting password");
    } finally {
      setLoading(false);
    }
  };

  const isDisabled = !password.trim() || !confirmPassword.trim();

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Reset Password for {userName}</DialogTitle>
        </DialogHeader>

        <div className="space-y-3">
          <Input
            type="password"
            placeholder="New password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <Input
            type="password"
            placeholder="Confirm new password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
          />
        </div>

        <DialogFooter className="flex justify-end gap-2">
          <Button
            variant="outline"
            onClick={onClose}
            className="cursor-pointer"
            disabled={loading}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            disabled={isDisabled || loading}
            className="bg-black hover:bg-gray-900 text-white flex items-center gap-2 cursor-pointer"
          >
            {loading ? (
              <>
                <span>Saving</span>
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