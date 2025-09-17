import React from 'react';
import { Bot } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';

interface ModelSelectorProps {
  value: string;
  onChange: (value: string) => void;
}

const availableModels = [
  { id: 'llama3:latest', name: 'Llama 3 (Latest)', description: 'Meta\'s latest Llama model' },
];

export const ModelSelector: React.FC<ModelSelectorProps> = ({ value, onChange }) => {
  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-2 text-muted-foreground">
        <Bot className="w-4 h-4" />
        <Label htmlFor="model-select" className="text-sm font-medium">
          Model:
        </Label>
      </div>
      
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger 
          id="model-select"
          className="w-48 bg-card border-border focus:border-primary focus:ring-primary/20"
        >
          <SelectValue placeholder="Select a model" />
        </SelectTrigger>
        <SelectContent>
          {availableModels.map((model) => (
            <SelectItem key={model.id} value={model.id}>
              <div className="flex flex-col items-start">
                <span className="font-medium">{model.name}</span>
                <span className="text-xs text-muted-foreground">{model.description}</span>
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
};