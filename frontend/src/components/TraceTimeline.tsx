import React from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Separator } from '@/components/ui/separator';
import { Sparkles, GitBranch, CornerDownRight, ChevronDown, ChevronUp, ListOrdered } from 'lucide-react';

type TraceEntry = {
  step: string;
  input: Record<string, any>;
  output: Record<string, any>;
};

interface TraceTimelineProps {
  trace: TraceEntry[];
}

const keyToLabel = (key: string) => key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

const JsonPills: React.FC<{ obj: Record<string, any> }>
  = ({ obj }) => {
  const entries = Object.entries(obj || {});
  if (entries.length === 0) return <span className="text-xs text-muted-foreground">None</span>;
  return (
    <div className="flex flex-wrap gap-2">
      {entries.map(([k, v]) => (
        <div key={k} className="px-2 py-1 rounded-md bg-card/60 border border-border text-xs">
          <span className="text-muted-foreground mr-1">{keyToLabel(k)}:</span>
          <span className="font-medium text-foreground">
            {typeof v === 'string' ? v : Array.isArray(v) ? JSON.stringify(v) : JSON.stringify(v)}
          </span>
        </div>
      ))}
    </div>
  );
};

export const TraceTimeline: React.FC<TraceTimelineProps> = ({ trace }) => {
  if (!trace || trace.length === 0) return null;

  const stepColors: Record<string, string> = {
    decompose_question: 'bg-secondary text-secondary-foreground',
    rewrite_query: 'bg-primary text-primary-foreground',
    generate_intermediate_answer: 'bg-purple-500 text-white',
    generate_final_answer: 'bg-success text-success-foreground',
  };

  const [expanded, setExpanded] = React.useState(false);

  const renderValue = (value: any) => {
    if (Array.isArray(value)) {
      return (
        <div className="w-full">
          <div className="flex items-center gap-2 mb-2 text-xs text-muted-foreground">
            <ListOrdered className="w-3 h-3" />
            <span>Items</span>
          </div>
          <ol className="list-decimal pl-4 space-y-1">
            {value.map((item, idx) => (
              <li key={idx} className="text-sm">
                {typeof item === 'string' ? (
                  <div className="inline-flex items-center gap-2">
                    <Badge variant="outline" className="bg-accent/20 border-accent/40 text-foreground">
                      {idx + 1}
                    </Badge>
                    <span>{item}</span>
                  </div>
                ) : (
                  <code className="text-xs bg-muted px-1 py-0.5 rounded">{JSON.stringify(item)}</code>
                )}
              </li>
            ))}
          </ol>
        </div>
      );
    }
    if (typeof value === 'string') return value;
    return <code className="text-xs bg-muted px-1 py-0.5 rounded">{JSON.stringify(value)}</code>;
  };

  const JsonPills: React.FC<{ obj: Record<string, any> }> = ({ obj }) => {
    const entries = Object.entries(obj || {});
    if (entries.length === 0) return <span className="text-xs text-muted-foreground">None</span>;
    return (
      <div className="space-y-3">
        {entries.map(([k, v]) => (
          <div key={k} className="">
            <div className="text-xs text-muted-foreground mb-1">{keyToLabel(k)}</div>
            {renderValue(v)}
          </div>
        ))}
      </div>
    );
  };

  return (
    <Card className="mt-4 p-4 bg-gradient-card border-border shadow-card">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">Reasoning Trace</h3>
        </div>
        <button
          onClick={() => setExpanded(e => !e)}
          className="text-xs inline-flex items-center gap-1 px-2 py-1 rounded border border-border hover:bg-accent/20"
        >
          {expanded ? (<><ChevronUp className="w-3 h-3" /> Collapse</>) : (<><ChevronDown className="w-3 h-3" /> Expand</>)}
        </button>
      </div>

      {!expanded && (
        <div className="text-xs text-muted-foreground">{trace.length} steps</div>
      )}

      {expanded && (
        <div className="relative">
          <div className="absolute left-3 top-0 bottom-0 w-0.5 bg-border" />
          <div className="space-y-3">
            {trace.map((t, idx) => (
              <div key={idx} className="relative pl-8">
                <div className="absolute left-0 top-2 w-6 h-6 rounded-full bg-background border border-border flex items-center justify-center">
                  <CornerDownRight className="w-3 h-3 text-muted-foreground" />
                </div>

                <Card className="p-3 bg-card/60 border-border">
                  <div className="flex items-center justify-between mb-2">
                    <Badge className={`${stepColors[t.step] || 'bg-accent text-accent-foreground'}`}>{t.step}</Badge>
                    <div className="text-xs text-muted-foreground flex items-center gap-1">
                      <Sparkles className="w-3 h-3" />
                      Step {idx + 1}
                    </div>
                  </div>

                  <Accordion type="multiple" defaultValue={["input","output"]}>
                    <AccordionItem value="input">
                      <AccordionTrigger className="text-sm">Input</AccordionTrigger>
                      <AccordionContent>
                        <JsonPills obj={t.input} />
                      </AccordionContent>
                    </AccordionItem>
                    <Separator className="my-2" />
                    <AccordionItem value="output">
                      <AccordionTrigger className="text-sm">Output</AccordionTrigger>
                      <AccordionContent>
                        <JsonPills obj={t.output} />
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                </Card>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
};

export default TraceTimeline;


