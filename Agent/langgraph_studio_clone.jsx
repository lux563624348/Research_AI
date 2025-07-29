// LangGraph Studio-like React frontend
import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { File, Folder } from "lucide-react";
import { motion } from "framer-motion";
import ReactFlow, { Background, Controls, MiniMap } from "reactflow";
import "reactflow/dist/style.css";

export default function LangGraphStudioClone() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [files, setFiles] = useState([]);
  const [elements, setElements] = useState([]);

  async function handleSubmit() {
    const res = await fetch("http://localhost:8123/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query })
    });
    const data = await res.json();
    setResponse(data.result || "No response");
    if (data.graph) setElements(data.graph); // optional: assumes {nodes, edges}
  }

  function handleFileChange(e) {
    const selected = Array.from(e.target.files);
    setFiles(selected);
  }

  return (
    <div className="grid grid-cols-3 gap-4 p-4">
      <div className="col-span-1 space-y-4">
        <Card>
          <CardContent className="space-y-2 p-4">
            <h2 className="text-xl font-bold">Ask LangGraph</h2>
            <Textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter query..."
              rows={6}
            />
            <Button onClick={handleSubmit} className="w-full">
              Run Query
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <h2 className="text-xl font-bold mb-2">Upload Files</h2>
            <Input type="file" multiple onChange={handleFileChange} />
            <div className="mt-2 max-h-40 overflow-y-auto">
              {files.map((file, idx) => (
                <div key={idx} className="flex items-center space-x-2">
                  <File className="w-4 h-4" />
                  <span className="text-sm">{file.name}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="col-span-2 space-y-4">
        <Card className="h-64">
          <CardContent className="p-4 overflow-auto">
            <h2 className="text-xl font-bold mb-2">Response</h2>
            <pre className="whitespace-pre-wrap text-sm text-muted-foreground">
              {response}
            </pre>
          </CardContent>
        </Card>

        <Card className="h-[500px]">
          <CardContent className="p-2 h-full">
            <h2 className="text-xl font-bold mb-2">Graph View</h2>
            <ReactFlow
              nodes={elements.nodes || []}
              edges={elements.edges || []}
              fitView
            >
              <MiniMap />
              <Controls />
              <Background gap={12} />
            </ReactFlow>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
