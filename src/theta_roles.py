import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import verbnet
import jsonlines
import json

class ThetaRoleAssigner:
    def __init__(self):
        """Initialize the theta role assignment system."""
        # Initialize VerbNet parser
        self.vn = verbnet()
        
    def get_verb_classes(self, verb):
        """
        Get VerbNet classes for a given verb.
        
        Args:
            verb: string, the verb to look up
            
        Returns:
            list of VerbNet class IDs
        """
        try:
            # Get all VerbNet classes containing this verb
            verb_classes = self.vn.get_verb_class_ids(verb)
            return verb_classes
        except:
            return []
    
    def get_theta_roles(self, verb_class_id):
        """
        Extract theta roles from a VerbNet class.
        
        Args:
            verb_class_id: string, VerbNet class identifier
            
        Returns:
            dict with frame information including theta roles
        """
        try:
            vnclass = self.vn.get_verb_class(verb_class_id)
            
            theta_info = {
                'class_id': verb_class_id,
                'members': [m.name for m in vnclass.members],
                'frames': []
            }
            
            # Extract frames and their thematic roles
            for frame in vnclass.frames:
                frame_data = {
                    'description': frame.description_num,
                    'primary': frame.primary,
                    'roles': []
                }
                
                # Get thematic roles from frame syntax
                for element in frame.syntax:
                    if hasattr(element, 'value'):
                        role_info = {
                            'type': element.value.get('type', ''),
                            'theta_role': element.value.get('value', '')
                        }
                        frame_data['roles'].append(role_info)
                
                theta_info['frames'].append(frame_data)
            
            return theta_info
        except:
            return None
    
    def get_wordnet_synsets(self, word, pos=None):
        """
        Get WordNet synsets for a word.
        
        Args:
            word: string, the word to look up
            pos: optional, part of speech (wn.VERB, wn.NOUN, etc.)
            
        Returns:
            list of synsets
        """
        if pos:
            return wn.synsets(word, pos=pos)
        return wn.synsets(word)
    
    def process_sentence(self, sentence, verb):
        """
        Process a sentence and assign theta roles based on the verb.
        
        Args:
            sentence: string, input sentence
            verb: string, the main verb to analyze
            
        Returns:
            dict with analysis results
        """
        result = {
            'sentence': sentence,
            'verb': verb,
            'verb_classes': [],
            'theta_assignments': []
        }
        
        # Get verb classes
        verb_classes = self.get_verb_classes(verb)
        result['verb_classes'] = verb_classes
        
        # Get theta role information for each class
        for vc in verb_classes:
            theta_info = self.get_theta_roles(vc)
            if theta_info:
                result['theta_assignments'].append(theta_info)
        
        # Get WordNet information
        synsets = self.get_wordnet_synsets(verb, pos=wn.VERB)
        result['wordnet_synsets'] = [s.name() for s in synsets[:3]]  # Top 3
        
        return result
    
    def process_jsonl(self, input_file, output_file):
        """
        Process a JSONL file with text input.
        
        Args:
            input_file: path to input .jsonl file
            output_file: path to output .jsonl file with annotations
        """
        with jsonlines.open(input_file) as reader, \
             jsonlines.open(output_file, mode='w') as writer:
            
            for obj in reader:
                # Assume each object has 'text' and 'verb' fields
                if 'text' in obj and 'verb' in obj:
                    result = self.process_sentence(obj['text'], obj['verb'])
                    obj['theta_analysis'] = result
                    writer.write(obj)


# Example usage
if __name__ == "__main__":
    assigner = ThetaRoleAssigner()
    
    # Example 1: Analyze a single verb
    print("=== Example 1: Verb 'give' ===")
    classes = assigner.get_verb_classes("give")
    print(f"VerbNet classes: {classes}")
    
    if classes:
        theta_info = assigner.get_theta_roles(classes[0])
        print(f"\nTheta roles for {classes[0]}:")
        print(json.dumps(theta_info, indent=2))
    
    # Example 2: Process a sentence
    print("\n=== Example 2: Sentence Analysis ===")
    result = assigner.process_sentence("John gave Mary a book", "give")
    print(json.dumps(result, indent=2))
    
    # Example 3: Process JSONL file
    # Uncomment to use:
    # assigner.process_jsonl('input.jsonl', 'output.jsonl')