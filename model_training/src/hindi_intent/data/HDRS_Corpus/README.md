 ## Hindi Dialogue Restaurant Search (HDRS) Corpus
 * This corpus is collected to promote research and development in the field of Hindi dialogue system.
 * Corpus consists of 1400 dialogues collected using Wizard-of-Oz fashion.
 * Corpus further divided into Training (800), Development (200) and Testing (400):
  * final_train_hin.json
  * final_dev_hin.json
  * final_test_hin.json

 * Its ontolody is defined as:

 | DA-Type | Slot | Total Number of Values |
 | :---: | :---: | :---: |
 | inform | food | 31 |
 | inform | area | 5 |
 | inform | price range | 3 |
 | request | address | - |
 | request | area | - |
 | request | food | - |
 | request | name | - |
 | request | phone | - |
 | request | postcode | - |

 * The corpus use the details of 118 Indian restaurants.
 * The structure of each dialogue in the corpus is as follows:
  * Dialogue Index: A unique index for dialogue identification.
  * Dialogue: It contains a collection of turns in a dialogue. A turn is a pair of system and user utterance. The information present in a turn contains:
    * Turn_index: A index value to indicate a turn in a dialogue uniquely
    * Transcript: The user utterance in written form.
    * Turn_Label: The DA corresponding to the current user utterance
    * Belief_state: The updated current state of the system. The state summarizes the history of the dialogue to provide ample details to choose the next move on the system. Therefore it maintains the DAs record.
    * System_transcript: The system utterance in written form. The system utterance in the corpus either conveys fetched information from the database or requests the user for more information.
    * System_act: The System DA corresponding to system utterance. The System DA is empty when the system utterance conveys fetched information from the database while it contains the requested slots when the system makes a request. These System DA are helpful when capturing the context of the previous turn.
