import numpy as np

from VisFly.utils.evaluate import TestBase
import os, sys
from typing import Optional
from matplotlib import pyplot as plt
from VisFly.utils.FigFashion.FigFashion import FigFon
import torch as th
import copy, cv2

from VisFly.utils.policies.td_policies import obs_as_tensor


class Test(TestBase):
    def __init__(self,
                 model,
                 name,
                 save_path: Optional[str] = None,
                 ):
        super(Test, self).__init__(model=model, name=name, save_path=save_path, )
        self.target_all = []
        self.target_dis_all = []
        self.center_all = []

    def draw(self, names=None):
        state_data = th.stack(self.state_all).cpu()
        targets_dis = th.stack(self.target_dis_all)
        action = th.stack([th.tensor(a) for a in self.action_all]).cpu()
        t = np.stack(self.t)[:, 0]
        aware_r = np.stack([r["aware_r"] for r in self.reward_all])
        percp_r = np.stack([r["percp_r"] for r in self.reward_all])
        for i in range(self.model.env.num_envs):
            fig = plt.figure(figsize=(9, 9))
            plt.subplot(4, 3, 1)
            plt.plot(t, state_data[:, i, 0:3], label=["x", "y", "z"])
            plt.legend()
            plt.subplot(4, 3, 2)

            plt.plot(t, state_data[:, i, 3:7], label=["w", "x", "y", "z"])
            plt.legend()
            plt.subplot(4, 3, 3)
            plt.plot(t, state_data[:, i, 7:10], label=["vx", "vy", "vz"])
            plt.legend()
            plt.subplot(4, 3, 4)
            plt.plot(t, state_data[:, i, 10:13], label=["wx", "wy", "wz"])
            plt.legend()
            plt.subplot(4, 3, 5)
            plt.plot(t[:-1], action[:, i, :], label=["a", "awx", "awy", "awz"])
            plt.legend()
            plt.subplot(4, 3, 6)
            plt.plot(t, targets_dis[:, i], label="target")
            plt.legend()
            plt.subplot(4, 3, 7)
            plt.plot(t[1:], percp_r[:,i], label="target")
            plt.title("perception reward")
            plt.subplot(4, 3, 8)
            plt.plot(t[1:], aware_r[:,i], label="target")
            plt.title("awareness reward")
            plt.subplot(4, 3, 9)
            col_dis = np.array([collision["col_dis"][i] for collision in self.collision_all])
            plt.plot(t, col_dis, label="closest distance")
            plt.title("closest distance")
            plt.subplot(4, 3, 10)
            plt.plot(t, state_data[:, i, 13:16], label=["x", "y", "z"])
            plt.title("acceleration")
            plt.subplot(4, 3, 11)
            plt.plot(t[:-1], (state_data[1:, i, 13:16] - state_data[:-1, i, 13:16]) / (t[2] - t[1]), label=["ax", "ay", "az"])
            plt.title("jerk")

            plt.tight_layout()
            plt.show()

        # col_dis = np.array([collision["col_dis"] for collision in self.collision_all])
        # fig2, axes = FigFon.get_figure_axes(SubFigSize=(1, 1))
        # axes.plot(t, col_dis)
        # axes.set_xlabel("t/s")
        # axes.set_ylabel("closest distance/m")
        plt.show()
        # print("rewards_sum: ", np_rewards)

        return [fig, ]

    def test(
            self,
            policy=None,
            world=None,
            # model=None,
            is_fig: bool = True,
            is_video: bool = True,
            is_sub_video: bool = True,
            is_fig_save: bool = True,
            is_video_save: bool = True,
            render_kwargs={},
            ROS_wrapper=None,
            debug=False,
            comment="",
    ):
        print(f"--------------------debug: enter test")

        # if is_fig_save:
        #     if not is_fig:
        #         raise ValueError("is_fig_save must be True if is_fig is True")

        if policy is None:
            policy = self.model.policy
        env = self.env
        if ROS_wrapper:
            ROS_env = ROS_wrapper(env, comment=comment)  # Replace env with ROS-wrapped version
            # ROS_env handles action/observation communication
            
        if debug:
            return env
        # done_all = th.full((env.num_envs,), False)
        print(f"--------------------debug: enter test and before reset")
        obs = env.reset(is_test=True)
        print(f"--------------------debug: enter test and finish reset")

        self._img_names = [name for name in obs.keys() if (("color" in name) or ("depth" in name) or ("semantic" in name))]
        if env.envs.dynamic_object_position[0][0] is not None:
            start_obj_pos = env.envs.dynamic_object_position[0].clone()
        self.obs_all.append(obs)
        if env.envs.dynamic_object_position[0][0] is not None:
            self.center_all.append(env.box_center.clone())
        self.state_all.append(env.extend_state.clone().detach())
        self.info_all.append([{} for _ in range(env.num_envs)])
        self.t.append(env.t.clone())
        self.target_dis_all.append((env.target - env.position).norm(dim=1))
        self.target_all.append((env.target))
        self.collision_all.append({"col_dis": env.collision_dis,
                                   "is_col": env.is_collision,
                                   "col_pt": env.collision_point})
        agent_index = [i for i in range(env.num_agent)]
        self.eq_r = []
        self.eq_l = []
        roun = 0
        prev_len = 0
        print("enter the loop")
        while True:
            with th.no_grad():
                if ROS_wrapper:
                    # For ROS wrapper, use its predict method which handles action communication
                    action = ROS_env.predict(obs, deterministic=True)
                else:
                    action = policy.predict(obs, deterministic=True)
                # action = policy.predict(obs, deterministic=True)
                if isinstance(action, tuple):
                    action = action[0]
                # obs, reward, done, info = env.step(action, is_test=True)
                if world is not None:
                    obs, reward, done, info = env.step(action, is_test=True, latent_func=world.step)
                else:
                    obs, reward, done, info = env.step(action, is_test=True)
                if ROS_wrapper:
                    ROS_env.publish_env_status()
                # = env.get_observation(), env.reward, env.done, env.info
                col_dis, is_col, col_pt = env.collision_dis, env.is_collision, env.collision_point
                state = env.state
                self.collision_all.append({"col_dis": col_dis, "is_col": is_col, "col_pt": col_pt})

            self.reward_all.append(copy.deepcopy(env._indiv_reward))
            self.action_all.append(action)
            self.state_all.append(env.extend_state.clone().detach())
            self.obs_all.append(obs)
            if env.envs.dynamic_object_position[0][0] is not None:
                self.center_all.append(env.box_center.clone())
            self.info_all.append(copy.deepcopy(info))
            self.target_dis_all.append((env.target - env.position).norm(dim=1))
            self.target_all.append((env.target))
            self.t.append(env.t.clone())
            if env.visual:
                # render_kwargs["points"] = th.atleast_2d(env.target)
                imgs = env.render(**render_kwargs)
                obs = obs_as_tensor(obs, device="cpu")
                if is_sub_video and len(self._img_names) > 0 and False:
                # add subvideo at right lower of the image
                    edge = 0.01
                    shape_img = imgs[0].shape[:2]
                    edge_int = int(min(shape_img) * edge)
                    # sub_image = (obs["depth"] /10 * 255).to(th.uint8).cpu().numpy()  # (N, C, H, W)
                    sub_image = (obs["color"]).to(th.uint8).cpu().numpy()  # (N, C, H, W)
                    sub_image = np.transpose(sub_image, (0, 2, 3, 1))  # (N, H, W, C)
                    # sub_image = np.tile(sub_image, (1, 1, 1, 3))  # (N, H, W, C)
                    # replace_dim = (shape_img[0] - sub_image.shape[1] - edge_int, shape_img[1])
                    replace_dim = (shape_img[0], shape_img[1] - sub_image.shape[1] - edge_int)
                    for i in range(len(obs["depth"])):
                        sub_image_shape = obs["depth"][i].shape[1:3]
                        # replace_dim = (
                        #     replace_dim[0],
                        #     replace_dim[1] - sub_image_shape[1] - edge_int
                        # )
                        replace_dim = (
                            replace_dim[0] - sub_image_shape[1] - edge_int,
                            replace_dim[1]
                        )
                        imgs[0][replace_dim[0]:(replace_dim[0] + sub_image_shape[0]),
                                replace_dim[1]:(replace_dim[1] + sub_image_shape[1]), :] = \
                            cv2.cvtColor(sub_image[i], cv2.COLOR_RGB2RGBA)

                render_image = cv2.cvtColor(imgs[0], cv2.COLOR_RGBA2RGB)

                self.render_image_all.append(render_image)
            # done_all[done] = True

            for i in reversed(agent_index):
                if done[i]:
                    self.eq_r.append(info[i]['episode']['r'].item())
                    self.eq_l.append(info[i]['episode']['l'].item())
                    agent_index.remove(i)

            if len(agent_index) == 0:
                break
            if env.envs.dynamic_object_position[0][0] is not None:
                if (start_obj_pos - env.envs.dynamic_object_position[0]).norm()<=0.2 and len(self.reward_all) > 30+prev_len:
                    roun += 1
                    prev_len = len(self.reward_all)

            print(len(self.reward_all), len(agent_index))
            if roun==1 and len(self.reward_all)>=300:
                break

        mean_r = th.as_tensor(self.eq_r, dtype=th.float32).mean().item()
        mean_l = th.as_tensor(self.eq_l, dtype=th.float32).mean().item()
        print(f"Average Rewards:{mean_r}, Average Length:{mean_l}")

        if is_fig:
            figs = self.draw()
            if is_fig_save:
                for i, fig in enumerate(figs):
                    self.save_fig(fig, c=i)
        else:
            figs = []
        if is_video:
            self.play(is_sub_video=is_sub_video)
        if is_video_save:
            self.save_video()

        render_video = th.as_tensor(np.stack(self.render_image_all, axis=0)).unsqueeze(0) if len(self.render_image_all) > 0 else None
        return figs, render_video, mean_r, mean_l
