# SentimentAnalysis
The application is a sentiment analysis application, which, given a piece of text, should be able to reply with its sentiment as being positive, negative, or neutral.

The result is a running application which the end user can access through a web browser, and start using immediately. The user is able to enter sentences in an input form and submit this sentence for a sentiment analysis. As a result, the analyzer will return either positive, negtive, or neutral as sentiment.

## Task Management

For our team internal task management, we use Trello. We decided to use four columns: **To Do**, **Doing**, **To be reviewed**, and **Done**. For each new task, the team member is free to move it to **Doing** and start on it. After he or she is finished, the task will be moved to **To be reviewed** for another team member to check it. If the reviewer accepts the changes, he or she moves the task to **Done**.

## Source Code Management

We use this GitHub repository to keep track of all our changes. For each task we work on, a new Branch is created with a clearly assignable branch name. At task completion, we create a pull request (PR) in GitHub and move the task in Trello to **To be reviewed**. The reviewer looks at the PR and, if he or she is happy with the made changes, merges the branch into the master branch.

## Testing

We aim at a high test coverage and want to have our code unit and integration tested. 

## Containerization

We want our application to be deliverable as a Docker image that contains the pre-trained model and the web interface.

## Running the application

The application needs a redis container to store the pretrained model and a flask container to serve the web interface. To start and connect both containers, we use a docker-compose file (see *docker-compose.yaml*). As a prerequisite, you need to have Docker installed on your machine.

To start the whole application, navigate to the folder in which the *docker-compose.yaml* can be found. From there, execute the following command:

```bash
docker-compose up
```

You can then visit the user interface locally under `http://0.0.0.0:5000/` or `http://localhost:5000/`. Enter a phrase and get the sentiment from there!
