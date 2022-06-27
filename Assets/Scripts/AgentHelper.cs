using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System;


// non state
public class AgentHelper : MonoBehaviour
{
    Settings settings;


    public AgentHelper(Settings settings)
    {
        this.settings = settings;
    }

    public bool AreAllAgentsOnTarget()
    {
        foreach (KeyValuePair<int, AgentInformation> entry in settings.AGENT_INFORMATION)
        {
            var agent_id = entry.Key;
            var agent_info = entry.Value;

            if (agent_info.agent.reached_goal == false)
            {
                return false;
            }
        }
        return true;
    }

    public void InstantiateAgents(Transform parent)
    {
        for (int row = 0; row < settings.MAP_HEIGHT; row++)
        {
            for (int col = 0; col < settings.MAP_WIDTH; col++)
            {
                var current_id = settings.MAP_AGENTS[row, col];

                // if agent exists on this tile
                if (Array.IndexOf(Settings.AGENTS_IDs, current_id) != -1)
                {
                    var tile = settings.TILES[row, col];
                    var agentPos = new Vector3(tile.posX, 1, tile.posZ);
                    var agentRot = Quaternion.identity;
                    var agentObj = Instantiate(settings.AGENT_PREFAB, agentPos, agentRot, parent);

                    var agent_information = settings.AGENT_INFORMATION[current_id];

                    var agent = agentObj.GetComponent<MyAgent>();
                    agent.id = current_id;
                    agent.order = agent_information.order;
                    agent.target_tile_id = agent_information.target_tile_id;
                    agent.onTile = tile;
                    agent.reached_goal = false;

                    var agentRenderer = agent.GetComponent<Renderer>();
                    agentRenderer.material.color = agent_information.color;

                    settings.AGENT_INFORMATION[current_id].agent = agent;

                    settings.AGENTS[row, col] = agent;

                }

            }
        }
    }

    public void ResetAgents(Transform parent)
    {
        for (int row = 0; row < settings.MAP_HEIGHT; row++)
        {
            for (int col = 0; col < settings.MAP_WIDTH; col++)
            {
                var current_id = settings.MAP_AGENTS[row, col];

                // if agent exists on this tile
                if (Array.IndexOf(Settings.AGENTS_IDs, current_id) != -1)
                {
                    var agent_information = settings.AGENT_INFORMATION[current_id];

                    var agent = agent_information.agent;
                    var agentObj = agent.gameObject;

                    var tile = settings.TILES[row, col];
                    var agentPos = new Vector3(tile.posX, 1, tile.posZ);
                    var agentRot = Quaternion.identity;
                    agentObj.transform.position = agentPos;
                    agentObj.transform.rotation = agentRot;

                    agent.id = current_id;
                    agent.order = agent_information.order;
                    agent.target_tile_id = agent_information.target_tile_id;
                    agent.reached_goal = false;
                    agent._reached_goal = false;
                    agent.onTile = tile;

                    var agentRenderer = agent.GetComponent<Renderer>();
                    agentRenderer.material.color = agent_information.color;

                    settings.AGENTS[row, col] = agent;
                }

            }
        }
    }



}
