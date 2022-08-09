package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

type archInfo struct {
	ArchitectureId      string           `json:"architecture_id"`
	Scores              map[string]SInfo `json:"scores"`
	TestAccuracy        string           `json:"test_accuracy"`
	TrainAccuracy       string           `json:"train_accuracy"`
	TrainableParameters string           `json:"trainable_parameters"`
	TrainingTime        string           `json:"training_time"`
	ValidationAccuracy  string           `json:"validation_accuracy"`
}

type Score struct {
	GradNorm       SInfo `json:"grad_norm"`
	Fisher         SInfo `json:"fisher"`
	GradPlain      SInfo `json:"grad_plain"`
	Grasp          SInfo `json:"grasp"`
	JacobConv      SInfo `json:"jacob_conv"`
	NasWot         SInfo `json:"nas_wot"`
	NtkCondNum     SInfo `json:"ntk_cond_num"`
	NtkTrace       SInfo `json:"ntk_trace"`
	NtkTraceApprox SInfo `json:"ntk_trace_approx"`
	Snip           SInfo `json:"snip"`
	Synflow        SInfo `json:"synflow"`
	WeightNorm     SInfo `json:"weight_norm"`
}

type SInfo struct {
	Score     string `json:"score"`
	TimeUsage string `json:"time_usage"`
}

func readJson(file_res string) map[string]archInfo {
	fileContent, err := os.Open(file_res)
	if err != nil {
		log.Fatal(err)
	}
	defer fileContent.Close()
	byteResult, _ := ioutil.ReadAll(fileContent)
	var res map[string]archInfo
	err = json.Unmarshal(byteResult, &res)
	if err != nil {
		fmt.Println(err)
	}
	return res
}

func compare_score_201_bn(m_name string, s1 float64, s2 float64) bool {
	if m_name == "grad_norm" {
		return s1 > s2
	}
	if m_name == "grad_plain" {
		return s1 < s2
	}
	if m_name == "ntk_cond_num" {
		return s1 < s2
	}
	if m_name == "ntk_trace" {
		return s1 > s2
	}
	if m_name == "ntk_trace_approx" {
		return s1 > s2
	}
	if m_name == "fisher" {
		return s1 > s2
	}
	if m_name == "grasp" {
		return s1 > s2
	}
	if m_name == "snip" {
		return s1 > s2
	}
	if m_name == "synflow" {
		return s1 > s2
	}
	if m_name == "weight_norm" {
		return s1 > s2
	}
	if m_name == "nas_wot" {
		return s1 > s2
	}
	return true
}

func compare_score_201_NO_bn(m_name string, s1 float64, s2 float64) bool {
	if m_name == "grad_norm" {
		return s1 < s2
	}
	if m_name == "grad_plain" {
		return s1 < s2
	}
	if m_name == "ntk_cond_num" {
		return s1 < s2
	}
	if m_name == "ntk_trace" {
		return s1 < s2
	}
	if m_name == "ntk_trace_approx" {
		return s1 < s2
	}
	if m_name == "fisher" {
		return s1 < s2
	}
	if m_name == "grasp" {
		return s1 > s2
	}
	if m_name == "snip" {
		return s1 < s2
	}
	if m_name == "synflow" {
		return s1 > s2
	}
	if m_name == "weight_norm" {
		return s1 < s2
	}
	if m_name == "nas_wot" {
		return s1 > s2
	}
	return true
}

func compare_score_101_bn(m_name string, s1 float64, s2 float64) bool {
	if m_name == "grad_norm" {
		return s1 < s2
	}
	if m_name == "grad_plain" {
		return s1 < s2
	}
	if m_name == "ntk_cond_num" {
		return s1 < s2
	}
	if m_name == "ntk_trace" {
		return s1 < s2
	}
	if m_name == "ntk_trace_approx" {
		return s1 < s2
	}
	if m_name == "fisher" {
		return s1 < s2
	}
	if m_name == "grasp" {
		return s1 > s2
	}
	if m_name == "snip" {
		return s1 < s2
	}
	if m_name == "synflow" {
		return s1 > s2
	}
	if m_name == "weight_norm" {
		return s1 > s2
	}
	if m_name == "nas_wot" {
		return s1 > s2
	}
	return true
}

func compare_score_101_NO_bn(m_name string, s1 float64, s2 float64) bool {
	if m_name == "grad_norm" {
		return s1 < s2
	}
	if m_name == "grad_plain" {
		return s1 < s2
	}
	if m_name == "ntk_cond_num" {
		return s1 < s2
	}
	if m_name == "ntk_trace" {
		return s1 < s2
	}
	if m_name == "ntk_trace_approx" {
		return s1 < s2
	}
	if m_name == "fisher" {
		return s1 < s2
	}
	if m_name == "grasp" {
		return s1 > s2
	}
	if m_name == "snip" {
		return s1 < s2
	}
	if m_name == "synflow" {
		return s1 > s2
	}
	if m_name == "weight_norm" {
		return s1 > s2
	}
	if m_name == "nas_wot" {
		return s1 > s2
	}
	return true
}

func compare_score_201_union_best(m_name string, s1 float64, s2 float64) bool {
	if m_name == "grad_norm" {
		return s1 > s2
	}
	if m_name == "grad_plain" {
		return s1 < s2
	}
	if m_name == "ntk_cond_num" {
		return s1 < s2
	}
	if m_name == "ntk_trace" {
		return s1 > s2
	}
	if m_name == "ntk_trace_approx" {
		return s1 > s2
	}
	if m_name == "fisher" {
		return s1 > s2
	}
	if m_name == "grasp" {
		return s1 > s2
	}
	if m_name == "snip" {
		return s1 > s2
	}
	if m_name == "synflow" {
		return s1 > s2
	}
	if m_name == "nas_wot" {
		return s1 > s2
	}
	// for combination, we assume bigger the better,
	return s1 > s2
}

func compare_score_101_union_best(m_name string, s1 float64, s2 float64) bool {
	if m_name == "grad_norm" {
		return s1 > s2
	}
	if m_name == "grad_plain" {
		return s1 < s2
	}
	if m_name == "ntk_cond_num" {
		return s1 < s2
	}
	if m_name == "ntk_trace" {
		return s1 > s2
	}
	if m_name == "ntk_trace_approx" {
		return s1 > s2
	}
	if m_name == "fisher" {
		return s1 > s2
	}
	if m_name == "grasp" {
		return s1 > s2
	}
	if m_name == "snip" {
		return s1 > s2
	}
	if m_name == "synflow" {
		return s1 > s2
	}
	if m_name == "nas_wot" {
		return s1 > s2
	}
	// for combination, we assume bigger the better,
	return s1 > s2
}

func return_better_arch(value1 archInfo, value2 archInfo, metricsList []string, spaceName string) string {
	var funcUsed func(m_name string, s1 float64, s2 float64) bool
	if spaceName == "101bn" {
		funcUsed = compare_score_101_bn
	} else if spaceName == "101nobn" {
		funcUsed = compare_score_101_NO_bn
	} else if spaceName == "201bn" {
		funcUsed = compare_score_201_bn
	} else if spaceName == "201nobn" {
		funcUsed = compare_score_201_NO_bn
	} else if spaceName == "201union_best" {
		funcUsed = compare_score_201_union_best
	} else if spaceName == "101union_best" {
		funcUsed = compare_score_101_union_best
	} else {
		fmt.Println("space cannot recognized")
	}
	leftVote := 0
	rightVote := 0
	for _, algName := range metricsList {
		var s1 float64
		var s2 float64
		s1, err1 := strconv.ParseFloat(value1.Scores[algName].Score, 64)
		s2, err2 := strconv.ParseFloat(value2.Scores[algName].Score, 64)
		if err1 != nil || err2 != nil {
			fmt.Println(algName, err2, err1)
			return err1.Error()
		}

		if funcUsed(algName, s1, s2) {
			leftVote += 1
		} else {
			rightVote += 1
		}
	}
	if leftVote > rightVote {
		return value1.ArchitectureId
	} else {
		return value2.ArchitectureId
	}
}

func allMetricsVoteTwoPair(data map[string]archInfo, spaceName string, voteList [][]string, fileName string, archList []string, thid int) {
	total_pair := 0
	vote_right := make(map[int]int)

	for index, _ := range voteList {
		vote_right[index] = 0
	}

	for _, archIDCombines := range archList {

		arch_list := strings.Split(archIDCombines, "__")

		archID1 := arch_list[0]
		archID2 := arch_list[1]
		value1 := data[archID1]
		value2 := data[archID2]
		value1.ArchitectureId = archID1
		value2.ArchitectureId = archID2

		if value1.TestAccuracy == value2.TestAccuracy {
			continue
		}

		for index, metricsList := range voteList {
			var v1_taccu float64
			var v2_taccu float64
			v1_taccu, err1 := strconv.ParseFloat(value1.TestAccuracy, 64)
			v2_taccu, err2 := strconv.ParseFloat(value2.TestAccuracy, 64)
			if err1 != nil || err2 != nil {
				fmt.Println(err1)
				fmt.Println(err2)
				return
			}
			if v1_taccu > v2_taccu && return_better_arch(value1, value2, metricsList, spaceName) == value1.ArchitectureId {
				vote_right[index] += 1
			}
			if v1_taccu < v2_taccu && return_better_arch(value1, value2, metricsList, spaceName) == value2.ArchitectureId {
				vote_right[index] += 1
			}
		}
		total_pair += 1
		//if total_pair%200000 == 0 {
		//	dt := time.Now()
		//	fmt.Printf("%s, threadID %d total_pair reaches %d \n", dt.String(), thid, total_pair)
		//}
	}
	filePath := fileName
	file, err := os.OpenFile(filePath, os.O_WRONLY|os.O_CREATE, 0666)
	if err != nil {
		fmt.Println("文件打开失败", err)
	}
	defer file.Close()
	write := bufio.NewWriter(file)
	write.WriteString(fmt.Sprintf("%s total_pair = %d \n", spaceName, total_pair))
	for index, ele := range vote_right {
		write.WriteString(fmt.Sprintf("%s, [%s], %d / %d \n", spaceName, strings.Join(voteList[index], ","), ele, total_pair))
	}
	write.Flush()
	write = nil
}

func toStr(m map[string]struct{}) string {
	var res string = "[]string{"
	for key, _ := range m {
		res += fmt.Sprintf("%s", key)
		res += ","
	}
	res += "}\n"
	return res
}

func getKeys(data map[string]archInfo) []string {
	var res []string
	for k, _ := range data {
		res = append(res, k)
	}
	return res

}

func getKeys2(data map[string]int) []string {
	var res []string
	for k, _ := range data {
		res = append(res, k)
	}
	return res
}

func readJsonArchIds(fileRes string) map[string]int {
	fileContent, err := os.Open(fileRes)
	if err != nil {
		log.Fatal(err)
	}
	defer fileContent.Close()
	byteResult, _ := ioutil.ReadAll(fileContent)
	var res map[string]int
	err = json.Unmarshal(byteResult, &res)
	if err != nil {
		fmt.Println(err)
	}
	return res
}

func measureAllVoteMetrics(
	space string,
	data201UnionBest map[string]archInfo,
	partiton_perfix string,
	totalCores int,
	allVoteMetrics [][]string,
	outputPrefix string,
) {

	var wg sync.WaitGroup
	wg.Add(totalCores)
	start0 := time.Now()
	for i := 0; i < totalCores; i++ {
		fileName := fmt.Sprintf("%s-%d", partiton_perfix, i)
		archPartitonMapper := readJsonArchIds(fileName)
		arch_combinations := getKeys2(archPartitonMapper)
		fmt.Printf("Run one thread computes %d archs \n", len(arch_combinations))
		go func(archs []string, iteIndexM int) {
			defer wg.Done()

			allMetricsVoteTwoPair(data201UnionBest,
				space,
				allVoteMetrics,
				fmt.Sprintf("%s/file-%d", outputPrefix, iteIndexM),
				archs, iteIndexM)
		}(arch_combinations, i)
	}
	fmt.Println("main thread wait here...")
	wg.Wait()
	fmt.Printf("201union, execution time %s\n", time.Since(start0))
}

func measureVoteList(space string, data map[string]archInfo, totalCores int, partition_perfix string, used_metrics [][]string, outputPrefix string) {

	var voteList [][]string
	if len(used_metrics) > 0 {
		voteList = used_metrics
	} else {
		for _, info := range data {
			for algName, _ := range info.Scores {
				if algName == "jacob_conv" || algName == "weight_norm" {
					continue
				}
				var inner []string
				inner = append(inner, algName)
				voteList = append(voteList, inner)
			}
			break
		}
	}

	//
	//grad_norm := []string{"grad_norm"}
	//grad_plain := []string{"grad_plain"}
	//ntk_cond_num := []string{"ntk_cond_num"}
	//ntk_trace := []string{"ntk_trace"}
	//ntk_trace_approx := []string{"ntk_trace_approx"}
	//fisher := []string{"fisher"}
	//grasp := []string{"grasp"}
	//snip := []string{"snip"}
	//synflow := []string{"synflow"}
	//weight_norm := []string{"weight_norm"}
	//nas_wot := []string{"nas_wot"}
	//
	//voteList := [][]string{
	//	grad_norm, grad_plain, ntk_cond_num, ntk_trace, ntk_trace_approx, grasp, fisher,
	//	snip, synflow, weight_norm, nas_wot}
	measureAllVoteMetrics(
		space,
		data,
		partition_perfix,
		totalCores, voteList, outputPrefix)
}

func main() {

	totalCores := 8

	runtime.GOMAXPROCS(totalCores)

	//spaceUsed := "101union_best"
	spaceUsed := "101union_best"
	dataFileName := "101_15625_c10_128_unionBest_with_vote.json"
	partitionFilePath := "101_correct_pair"
	dataFolderName := "CIFAR10_15625"

	baseFolder := "/home/naili/Fast-AutoNAS/result/"
	//baseFolder := "/Users/kevin/project_python/Fast-AutoNAS/result/"

	datafilePath := filepath.Join(baseFolder, dataFolderName, "vote_res", dataFileName)
	partitionPrefix := filepath.Join(baseFolder, "space_partition", partitionFilePath, "partition")

	outputPrefix := filepath.Join(baseFolder, dataFolderName, partitionFilePath)

	data201UnionBest := readJson(datafilePath)

	// 	votes := [][]string{{"nas_wot", "snip", "synflow"}}
	var votes [][]string

	measureVoteList(
		spaceUsed,
		data201UnionBest,
		totalCores,
		partitionPrefix,
		votes, outputPrefix)
}
