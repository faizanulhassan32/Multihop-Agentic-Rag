import React from 'react';
import { Zap, AlertCircle, CheckCircle2, Loader2, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

interface StatusBarProps {
  langGraphStatus: 'checking' | 'connected' | 'disconnected';
  onRefreshStatus: () => void;
}

export const StatusBar: React.FC<StatusBarProps> = ({
  langGraphStatus,
  onRefreshStatus,
}) => {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected':
        return <CheckCircle2 className="w-4 h-4 text-success" />;
      case 'disconnected':
        return <AlertCircle className="w-4 h-4 text-destructive" />;
      case 'creating':
        return <Loader2 className="w-4 h-4 text-warning animate-spin" />;
      default:
        return <Loader2 className="w-4 h-4 text-muted-foreground animate-spin" />;
    }
  };

  const getStatusText = (service: string, status: string) => {
    switch (status) {
      case 'connected':
        return `${service} Connected`;
      case 'disconnected':
        return `${service} Disconnected`;
      case 'creating':
        return `Creating ${service}...`;
      default:
        return `Checking ${service}...`;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected':
        return 'text-success';
      case 'disconnected':
        return 'text-destructive';
      case 'creating':
        return 'text-warning';
      default:
        return 'text-muted-foreground';
    }
  };

  return (
    <Card className="m-4 mb-0 p-4 bg-gradient-card border-border shadow-card">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-6">
          {/* LangGraph Status Only */}
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-secondary" />
            <div className="flex items-center gap-2">
              {getStatusIcon(langGraphStatus)}
              <span className={`text-sm font-medium ${getStatusColor(langGraphStatus)}`}>
                {getStatusText('LangGraph', langGraphStatus)}
              </span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button
            onClick={onRefreshStatus}
            variant="outline"
            size="sm"
            className="border-border hover:border-primary/50"
          >
            <RefreshCw className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </Card>
  );
};