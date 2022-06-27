using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TargetTile : Tile
{
    public TargetTile(int row, int col, float posX, float posY, float posZ) : base(row, col, posX, posY, posZ)
    {
    }


    private void OnCollisionEnter(Collision collision)
    {
        /*if (collision.collider.CompareTag("agent"))
        {
            var agentObj = collision.collider.GetComponent<MyAgent>();
            agentObj.SetReward(1);
            agentObj.EndEpisode();

            agentObj.StartColorChangeCoroutine(0.3f);

            //StartCoroutine(ChangeTileColor(agentObj, 0.2f));
        }*/
    }

    // will not work because I destroy the tile right after the epsiode ends
    IEnumerator ChangeTileColor(MyAgent agent, float time)
    {
        var tileRenderer = agent.GetComponent<Renderer>();
        var oldColor = tileRenderer.material.color;
        tileRenderer.material.color = Color.yellow;
        yield return new WaitForSeconds(time); // Wait for 2 sec
        tileRenderer.material.color = Color.red;
    }
}
