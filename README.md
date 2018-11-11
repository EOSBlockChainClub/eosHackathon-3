# eosHackathon

### Brief description of our project:

#### Front End Website
    - The user will be asked to enter their username, first name, and last name, and age.
    - The username should be unique for each user, and the take picture button will prompt the user to turn on their camera and will capture a snap of the user when they are clearly facing the camera.
    - After the user's image has been snapped, all of their data will be sent to the blockchain.

#### Back End/Blockchain
    - Generate the smart contract, which allows the user and the company (data buyers) to enter an agreement.
    - Once the user gives the company permission to access their data, tokens are sent to the customers' public wallet addresses.
    - Contains all the information about the user. This includes the data to be sent to the blockchain.

#### Facial Recognition Part
    - Facial recognition system uses powerful open source open CV library on python. 
    - Test2 is a tool to store a person photo in the library. 
    - Face_Recognizer is the main class that can take number of users and train a AI model to classify a given picture among users. 
    - In future it is planned to implement transfer learning algorithm from Face Net model because it is going to more accurate algorithm. 
    - Also we plan to implement PyGaze open source library to track a eye movements of a user to prevent fraud with facial recognition system
    - Test5 is implemented to actually recognize the user’s user name
