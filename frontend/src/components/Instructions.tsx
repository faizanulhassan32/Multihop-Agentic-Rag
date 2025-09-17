import React from 'react';
import { Info, CheckCircle, Play } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

export const Instructions: React.FC = () => {
  return (
    <Card className="p-6 bg-gradient-card border-border shadow-card">
      <div className="flex items-center gap-2 mb-4">
        <Info className="w-5 h-5 text-primary" />
        <h3 className="font-semibold text-foreground">Getting Started</h3>
      </div>

      <div className="space-y-4">
        <div className="space-y-3">
          <div className="flex items-start gap-3">
            <Badge variant="outline" className="bg-primary/10 text-primary border-primary/20">
              1
            </Badge>
            <div className="flex-1">
              <p className="text-sm text-muted-foreground">
                Make sure your <code className="bg-muted px-1.5 py-0.5 rounded text-xs">GOOGLE_API_KEY</code> is set in your environment.
              </p>
            </div>
          </div>

          <div className="flex items-start gap-3">
            <Badge variant="outline" className="bg-secondary/10 text-secondary border-secondary/20">
              2
            </Badge>
            <div className="flex-1">
              <p className="text-sm text-muted-foreground">
                Start the LangGraph server in your terminal:
              </p>
              <div className="bg-muted p-2 rounded mt-1">
                <code className="text-xs flex items-center gap-1">
                  <Play className="w-3 h-3" />
                  langgraph dev
                </code>
              </div>
            </div>
          </div>

          <div className="flex items-start gap-3">
            <Badge variant="outline" className="bg-success/10 text-success border-success/20">
              3
            </Badge>
            <div className="flex-1">
              <p className="text-sm text-muted-foreground">
                Make sure that the HotpotQA database is already established.
              </p>
            </div>
          </div>

          <div className="flex items-start gap-3">
            <Badge variant="outline" className="bg-warning/10 text-warning border-warning/20">
              4
            </Badge>
            <div className="flex-1">
              <p className="text-sm text-muted-foreground">
                Ask a complex multi-hop question or use one of the samples above.
              </p>
            </div>
          </div>
        </div>

        <div className="border-t border-border pt-4">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <CheckCircle className="w-4 h-4 text-success" />
            <span>Multi-hop reasoning enabled for complex questions</span>
          </div>
        </div>
      </div>
    </Card>
  );
};