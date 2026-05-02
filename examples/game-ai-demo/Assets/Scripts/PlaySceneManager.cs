using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

/// <summary>
/// Play scene - allows human player to play against evolved AI
/// </summary>
public class PlaySceneManager : MonoBehaviour
{
    [SerializeField] private GameObject playerPrefab;
    [SerializeField] private GameObject aiPrefab;
    [SerializeField] private Text scoreText;
    [SerializeField] private Text levelText;
    [SerializeField] private Button backButton;
    
    private int playerScore = 0;
    private int aiScore = 0;
    private int currentLevel = 1;
    
    private void Start()
    {
        SpawnGameObjects();
        
        if (backButton != null)
        {
            backButton.onClick.AddListener(() => SceneManager.LoadScene("TrainingScene"));
        }
    }
    
    private void SpawnGameObjects()
    {
        // Spawn player (usually controlled by human)
        if (playerPrefab != null)
        {
            Instantiate(playerPrefab, Vector3.zero, Quaternion.identity);
        }
        
        // Spawn evolved AI opponent
        if (aiPrefab != null)
        {
            Instantiate(aiPrefab, new Vector3(5, 0, 5), Quaternion.identity);
        }
    }
    
    private void Update()
    {
        UpdateUI();
    }
    
    private void UpdateUI()
    {
        if (scoreText != null)
            scoreText.text = $"Score: Player {playerScore} - AI {aiScore}";
        if (levelText != null)
            levelText.text = $"Level: {currentLevel}";
    }
    
    public void IncrementAIScore()
    {
        aiScore++;
    }
    
    public void IncrementPlayerScore()
    {
        playerScore++;
    }
}
