import React from 'react';
import { HelpCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

interface SampleQuestionsProps {
  onQuestionSelect: (question: string) => void;
}

const sampleQuestions = [
  "In which year was the actor who played Harry Potter born?",
  "What is the population of the city where Harvard University is located?",
  "Who is the author of the book that the movie 'The Shawshank Redemption' is based on?",
  "What genre of music is performed by the band that released the album 'Dark Side of the Moon'?",
  "What is the capital of the country where Mount Everest is located?",
];

export const SampleQuestions: React.FC<SampleQuestionsProps> = ({ onQuestionSelect }) => {
  return (
    <Card className="p-6 bg-gradient-card border-border shadow-card">
      <div className="flex items-center gap-2 mb-4">
        <HelpCircle className="w-5 h-5 text-secondary" />
        <h3 className="font-semibold text-foreground">Sample Questions</h3>
      </div>
      
      <div className="grid gap-2">
        {sampleQuestions.map((question, index) => (
          <Button
            key={index}
            variant="outline"
            className="text-left h-auto p-3 justify-start border-border hover:border-primary/50 hover:bg-primary/5 transition-all duration-200"
            onClick={() => onQuestionSelect(question)}
          >
            <span className="text-sm text-muted-foreground mr-2">{index + 1}.</span>
            <span className="text-sm">{question}</span>
          </Button>
        ))}
      </div>
      
      <p className="text-xs text-muted-foreground mt-4 text-center">
        Click any question to automatically fill the input field
      </p>
    </Card>
  );
};