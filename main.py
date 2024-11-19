from classes import *

freq_eval = 250000
reward_threshold = 3
num_episodes = 1000
best_model = "random"
reset_timesteps = True

# result = []
# for _ in range(0, 1000):
#     score = model_fight(old_model, test_model)
#     result.append(score)

num_champions = 0

while True:

    champion = 0

    env = SkyjoEnv(best_model=best_model)
    # It will check your custom environment and output additional warnings if needed
    check_env(env)

    env = ActionMasker(env, mask_fn)

    model = MaskablePPO(MaskableMultiInputActorCriticPolicy,
                        env,
                        verbose=1,
                        tensorboard_log="./first_test_skyjo",
                        learning_rate=0.0002,
                        n_epochs=15,
                        gamma=0.995,
                        clip_range=0.1,
                        batch_size=128)

    while not champion:
        print("Actual best model : ")
        print(env.best_model)
        model.learn(freq_eval, reset_num_timesteps=reset_timesteps)

        print("------------------- NEW MODEL FIGHTING ! -------------------")
        result_model = []
        for _ in range(0, num_episodes):
            score = model_fight(model, env.best_model)
            result_model.append(score)

        print(f"Result new model vs ancien model : {np.mean(result_model)}")

        if np.mean(result_model) > reward_threshold:
            print("We have a new champion !")
            best_model = model
            model.save(f"./models/Skyjo_champion_{num_champions}")
            champion = 1
            num_champions += 1
            reset_timesteps = True
        else:
            print("You'll do better next time, little agent")
            reset_timesteps = False

# model.set_env(env)

# for i in range(0, 50):
#     if i == 0:
#         model = MaskablePPO(MaskableMultiInputActorCriticPolicy,
#                             env,
#                             verbose=1,
#                             tensorboard_log="./first_test_skyjo",
#                             learning_rate=0.0002,
#                             n_epochs=15,
#                             gamma=0.995,
#                             clip_range=0.1,
#                             batch_size=128)
#         model.learn(total_timesteps=1000000)
#         model.save(f"./models/Skyjo_v0.2.{i}")
#     else:
#         model = MaskablePPO.load(f"./models/Skyjo_v0.2.{i-1}")
#         model.set_env(env)
#         model.learn(total_timesteps=1000000, reset_num_timesteps=False)
#         model.save(f"./models/Skyjo_v0.2.{i}")