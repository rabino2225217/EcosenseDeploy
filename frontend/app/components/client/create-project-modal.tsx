import * as React from "react";
import { Plus, Loader } from "lucide-react";
import { Button } from "../ui/button";
import { Label } from "../ui/label";
import { Input } from "../ui/input";
import { Textarea } from "../ui/textarea";
import { toast } from "sonner";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "../ui/select";

const BLISST = ["Baguio", "La Trinidad", "Itogon", "Sablan", "Tuba", "Tublay"];

export type NewProjectPayload = {
  name: string;
  location: string;
  description?: string;
};

export type CreateResult = { success: boolean; message: string };

export default function Modal({
  onCreate,
  onError,
}: {
  onCreate?: (data: NewProjectPayload) => Promise<CreateResult>;
  onError?: (msg: string) => void;
}) {
  const [open, setOpen] = React.useState(false);
  const [name, setName] = React.useState("");
  const [location, setLocation] = React.useState<string>("");
  const [description, setDescription] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const maxLength = 50;

  React.useEffect(() => {
    if (open) {
      setName("");
      setLocation("");
      setDescription("");
      onError?.("");
    }
  }, [open]);

  React.useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [open]);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();

    if (!name.trim() || !location) {
      const msg = "Please provide a name and choose a location.";
      toast.error(msg);
      onError?.(msg);
      return;
    }

    if (name.trim().length > maxLength) {
      toast.error(`Project name cannot exceed ${maxLength} characters.`);
      return;
    }

    setLoading(true);
    try {
      const { success, message } = (await onCreate?.({
        name: name.trim(),
        location,
        description: description.trim() || undefined,
      })) || { success: false, message: "Something went wrong." };

      if (success) {
        toast.success(message);
        setName("");
        setLocation("");
        setDescription("");
        setOpen(false);
        onError?.("");

        window.dispatchEvent(new CustomEvent("qg:create-project-done"));
      } else {
        toast.error(message);
        onError?.(message);
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <Button
        type="button"
        className="gap-2 rounded-md cursor-pointer hover:bg-emerald-400"
        data-qg="new-project"
        onClick={() => setOpen(true)}
      >
        <Plus className="h-4 w-4" />
        New
      </Button>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center"
          role="dialog"
          aria-modal="true"
        >
          <div
            className="absolute inset-0 bg-black/40"
            onClick={() => setOpen(false)}
          />

          <div
            className="relative z-10 w-[92vw] max-w-md rounded-2xl bg-white p-6 shadow-xl dark:bg-neutral-900"
            onClick={(e) => e.stopPropagation()}
            data-qg="create-project-modal"
          >
            <h2 className="text-xl font-semibold">New Project</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Fill out the project details below.
            </p>

            {/* FORM */}
            <form onSubmit={handleSubmit} className="mt-5 space-y-4">
              {/* Name */}
              <div className="space-y-1.5">
                <Label htmlFor="proj-name">Name</Label>
                <Input
                  id="proj-name"
                  placeholder="e.g. Tree Mapping Project"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  maxLength={maxLength}
                  required
                />
                <p className="text-xs text-muted-foreground">
                  {name.length}/{maxLength}
                </p>
              </div>

              {/* Location */}
              <div className="space-y-1.5">
                <Label>Location (BLISST)</Label>
                <Select value={location} required onValueChange={setLocation}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select location" />
                  </SelectTrigger>
                  <SelectContent>
                    {BLISST.map((l) => (
                      <SelectItem key={l} value={l}>
                        {l}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Description */}
              <div className="space-y-1.5">
                <Label htmlFor="proj-desc">Description</Label>
                <Textarea
                  id="proj-desc"
                  placeholder="Optional notes about this project…"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="min-h-[110px]"
                />
              </div>

              {/* Buttons */}
              <div className="mt-4 flex justify-end gap-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setOpen(false)}
                  disabled={loading}
                  className="cursor-pointer"
                >
                  Cancel
                </Button>

                <Button
                  type="submit"
                  disabled={loading}
                  className="cursor-pointer flex items-center gap-2"
                >
                  {loading ? (
                    <>
                      Creating
                      <Loader className="h-4 w-4 animate-spin" />
                    </>
                  ) : (
                    "Create"
                  )}
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
}
