# 🚀 Dating App Simulator: Diving into Modern Dating Apps

Dynamic Modeling of Social Interactions in Dating App Environments

## 🎯 Project Objectives

- 🕵️‍♀️ Uncover hidden patterns in the world of online dating
- 🌟 Explore how factors like attractiveness, gender ratios, and user preferences shape dating dynamics
- 📊 Compare our simulated results with real-world dating data
- 😤 Investigate the roots of user frustration in dating apps

## 🧠 The Science Behind the Swipes

This project uses reinforcement learning to simulate various dating environments. Matchmaking and attraction aren't concepts AI can ever possibly simulate, but we hope to use AI as a tool to understand the complexities of online dating.

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.7+ installed. Then, install the required packages:

```bash
pip install -r requirements.txt
```

### Running Simulations

To start playing Cupid with AI, run:

```bash
python simulation.py
```

This will run all the simulations defined in `game_config/multi_sim_config.json`.

## 📊 Simulation Types

1. **Basic 50-50 Model**: A simple world where love is truly blind (and random)!
2. **Gender Imbalance Model**: Exploring what happens when the dating pool isn't evenly split
3. **Multiple Attributes Model**: Because we're all more than just a pretty face
4. **Advanced Model**: The kitchen sink of dating simulations - includes everything but the actual date

## 🛠️ Customization

Want to create your own dating dystopia (or utopia)? Modify the `game_config/multi_sim_config.json` file to adjust:

- Population sizes
- Attribute distributions
- Swiping behaviors
- And more!

If you're interested in checking out what the results from our simulations were, please view our Final Report write-up which contains a thorough overview of this project as a whole.

## 📈 Visualizing Results

After each simulation, our simulation will display:

- Cumulative matches over time
- Average match quality
- Match rate for both genders

See how small tweaks in your starting configuration can produce drastically different outcomes.

## 🤝 Contributing

Found a bug? Want to add a new feature? We're all about that commitment! Feel free to open an issue or submit a pull request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🙏 Acknowledgments

- Shoutout to the Primer YouTube channel for inspiring our simulations
- Thank you to our wonderful DS4420 Professor Deahan Yu for his guidance and expertise
- Acknowledgement of all the resources cited in our paper 
