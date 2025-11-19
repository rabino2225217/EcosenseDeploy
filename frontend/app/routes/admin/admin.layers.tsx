import socket from "../../services/socket";
import { useEffect, useState } from "react";
import { Search, Loader } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card";

type Layer = {
  name: string;
  isActive: boolean;
};

const API_URL = import.meta.env.VITE_API_URL;

export default function AdminLayers() {
  const [layers, setLayers] = useState<Layer[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [loadingLayer, setLoadingLayer] = useState<string | null>(null);

  //Fetch all layers and active layer
  useEffect(() => {
    const fetchLayers = async () => {
      setLoading(true);
      try {
        const res = await fetch(`${API_URL}/admin/wms/capabilities`);
        const data: { layers: { name: string }[] } = await res.json();

        const activeRes = await fetch(`${API_URL}/admin/wms/active-layer`);
        const activeData: { layerName: string | null } = activeRes.ok
          ? await activeRes.json()
          : { layerName: null };

        setLayers(
          data.layers.map((l) => ({
            name: l.name,
            isActive: activeData.layerName === l.name,
          }))
        );
      } catch (err) {
        console.error("Error fetching layers:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchLayers();
  }, []);

  //Socket connection
  useEffect(() => {
    socket.on("active-layer-updated", (data: { layerName: string }) => {
      setLayers((prev) =>
        prev.map((l) => ({ ...l, isActive: l.name === data.layerName }))
      );
    });

    return () => {
      socket.off("active-layer-updated");
    };
  }, []);

  //Set active layer
  const setActiveLayer = async (layer: Layer) => {
    setLoadingLayer(layer.name);
    try {
      await fetch(`${API_URL}/admin/wms/save-layer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ layerName: layer.name }),
      });
    setLayers(prev =>
      prev.map(l => ({ ...l, isActive: l.name === layer.name }))
    );  
    } catch (err) {
      console.error("Error setting active layer:", err);
    }
  };

  const filtered = layers.filter((l) =>
    l.name.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-2xl font-bold text-gray-800">Manage Maps</h3>

        {/* Search Input */}
        <div className="relative">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search layers..."
            className="h-10 w-64 rounded-md border bg-white pl-10 pr-3 text-sm text-black outline-none focus:ring-2 focus:ring-emerald-300"
          />
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
        </div>
      </div>

      {/* Table */}
      <Card>
        <CardHeader>
          <CardTitle>Available Layers</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-hidden rounded-xl border border-gray-200 shadow-sm">
            <table className="w-full text-sm text-gray-700">
              <thead>
                <tr className="bg-gray-100 text-left">
                  <th className="px-6 py-3 font-semibold">Layer Name</th>
                  <th className="px-6 py-3 font-semibold text-center">Status</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={2} className="text-center p-6 text-gray-500">
                      <span className="flex items-center justify-center gap-2">
                        Loading layers
                        <Loader className="w-5 h-5 animate-spin" />
                      </span>
                    </td>
                  </tr>
                ) : filtered.length === 0 ? (
                  <tr>
                    <td colSpan={2} className="text-center p-6 text-gray-500">
                      No layers found.
                    </td>
                  </tr>
                ) : (
                  filtered.map((layer) => (
                    <tr
                      key={layer.name}
                      className="bg-white transition border-b border-gray-200 hover:bg-gray-50"
                    >
                      <td className="px-6 py-3">{layer.name}</td>
                      <td className="px-6 py-3 text-center">
                        {layer.isActive ? (
                          <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-emerald-100 text-emerald-700">
                            Active
                          </span>
                        ) : (
                          <button
                            onClick={() => setActiveLayer(layer)}
                            disabled={loadingLayer === layer.name}
                            className="inline-flex items-center justify-center px-3 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-600 hover:bg-emerald-50 hover:text-emerald-700 transition cursor-pointer disabled:opacity-50"
                          >
                            {loadingLayer === layer.name ? (
                              <Loader className="w-4 h-4 animate-spin" />
                            ) : (
                              "Set Active"
                            )}
                          </button>
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}